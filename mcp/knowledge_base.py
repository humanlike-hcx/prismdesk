"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 文档导入：将文本切片，通过配置的服务生成 Embedding 后存入 ChromaDB
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 与 MCP 工具框架集成：作为 knowledge_search 工具的真实 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储知识库文档（RAG 检索）
  两者是不同的 collection，互不干扰。
"""
import asyncio
import hashlib
import logging
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
import httpx

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    生产模式通过 OpenAI 兼容的 Embedding API 显式生成文档和查询向量，
    ChromaDB 只负责存储与相似度检索。文档和查询必须使用同一模型。
    未配置远端 Embedding 时保留 Chroma 默认模型，便于离线开发。
    """

    DEFAULT_COLLECTION_NAME = "knowledge_base"

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
        embedding_api_key: str = "",
        embedding_base_url: str = "",
        embedding_model: str = "",
        embedding_timeout_s: float = 20.0,
        collection_name: str = "",
    ):
        self._embedding_api_key = embedding_api_key.strip()
        base_url = embedding_base_url.strip().rstrip("/")
        self._embedding_endpoint = base_url if base_url.endswith("/embeddings") else f"{base_url}/embeddings"
        self._embedding_model = embedding_model.strip()
        self._embedding_timeout_s = embedding_timeout_s
        self._remote_embedding_enabled = bool(
            self._embedding_api_key and base_url and self._embedding_model
        )
        configured_name = collection_name.strip() or os.getenv("RAG_COLLECTION_NAME", "").strip()
        self.collection_name = configured_name or self.DEFAULT_COLLECTION_NAME
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{1,510}[A-Za-z0-9]", self.collection_name):
            raise ValueError(f"非法 RAG collection 名称: {self.collection_name!r}")
        # 优先连接独立 ChromaDB 服务；Embedding 由当前 Python 调用侧负责生成。
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception:
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # Chroma 的 HttpClient 仍在调用侧生成 query_texts 的向量；远端模式
        # 因此始终显式传 embeddings，避免 CLI 下载默认 ONNX 模型。
        collection_metadata = {
            "description": "PrismDesk RAG 知识库",
            "embedding_provider": "remote" if self._remote_embedding_enabled else "chroma_default",
            "embedding_model": self._embedding_model or "all-MiniLM-L6-v2",
            "hnsw:space": "cosine",
        }
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata=collection_metadata,
        )

        # 如果知识库为空，默认导入演示文档。演示启动可显式关闭，避免首次
        # 下载 Chroma 默认 Embedding 模型阻塞 API 就绪；真实业务应通过
        # /knowledge/add 或 /knowledge/upload 导入自己的资料。
        if self._collection.count() == 0:
            load_defaults = os.getenv("PRISMDESK_LOAD_DEFAULT_KB", "true").strip().lower()
            if load_defaults in {"1", "true", "yes", "on"}:
                self._load_default_docs()
            else:
                logger.info("知识库为空，已跳过默认演示文档初始化")

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式: [{"title": "...", "content": "..."}, ...]
        长文档会自动切片（每片 500 字）。
        """
        ids, docs, metas = [], [], []

        for doc in documents:
            title   = doc.get("title", "")
            content = doc.get("content", "")
            chunks  = self._chunk_text(content, chunk_size=500)

            for i, chunk in enumerate(chunks):
                doc_id = hashlib.md5(f"{title}_{i}_{chunk[:50]}".encode()).hexdigest()
                ids.append(doc_id)
                docs.append(chunk)
                metas.append({"title": title, "chunk_index": i, "total_chunks": len(chunks)})

        if ids:
            # 远端模式显式生成向量；不允许失败后静默切换模型，否则同一
            # collection 会混入不同维度或不同向量空间的数据。
            embeddings = self._embed_texts(docs) if self._remote_embedding_enabled else None
            kwargs: Dict[str, Any] = {"ids": ids, "documents": docs, "metadatas": metas}
            if embeddings is not None:
                kwargs["embeddings"] = embeddings
            self._collection.upsert(**kwargs)
            logger.info(f"知识库导入 {len(ids)} 个文档片段")

        return len(ids)

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        语义检索：根据 query 返回最相关的文档片段。

        远端模式先用配置的模型生成 query 向量，再由 ChromaDB 做余弦匹配。
        """
        # 空库没有可检索内容，直接返回，避免首次请求为无意义的 query 下载
        # 默认 Embedding 模型并阻塞用户响应。
        if self._collection.count() == 0:
            return []
        if self._remote_embedding_enabled:
            results = self._collection.query(
                query_embeddings=self._embed_texts([query]),
                n_results=top_k,
            )
        else:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
            )

        items = []
        if results["documents"] and results["documents"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                items.append({
                    "title":    meta.get("title", ""),
                    "content":  doc,
                    "score":    round(1.0 - dist, 4),  # ChromaDB 返回距离，转为相似度
                    "chunk":    meta.get("chunk_index", 0),
                })

        return items

    async def search_async(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k)

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """调用 OpenAI 兼容接口；只返回按 input 顺序排列的向量。"""
        if not self._remote_embedding_enabled:
            raise RuntimeError("远端 Embedding 未配置")
        headers = {
            "Authorization": f"Bearer {self._embedding_api_key}",
            "Content-Type": "application/json",
        }
        try:
            with httpx.Client(timeout=self._embedding_timeout_s) as client:
                response = client.post(
                    self._embedding_endpoint,
                    headers=headers,
                    json={"model": self._embedding_model, "input": texts},
                )
                response.raise_for_status()
                data = response.json().get("data", [])
            vectors = [
                list(item["embedding"])
                for item in sorted(data, key=lambda item: int(item.get("index", 0)))
            ]
            if len(vectors) != len(texts) or any(not vector for vector in vectors):
                raise ValueError("返回的向量数量或内容不正确")
            dimensions = {len(vector) for vector in vectors}
            if len(dimensions) != 1:
                raise ValueError("返回的向量维度不一致")
            return vectors
        except Exception as ex:
            raise RuntimeError(f"RAG Embedding 调用失败: {ex}") from ex

    @property
    def embedding_info(self) -> Dict[str, Any]:
        return {
            "provider": "remote" if self._remote_embedding_enabled else "chroma_default",
            "model": self._embedding_model or "all-MiniLM-L6-v2",
            "collection": self.collection_name,
        }

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        return await self.search_async(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """从版本化知识包导入演示环境客服规范。"""
        knowledge_path = Path(__file__).resolve().parent.parent / "knowledge" / "standard_customer_service.json"
        default_docs = json.loads(knowledge_path.read_text(encoding="utf-8"))
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
