"""三路业务路由识别。

识别器只回答一个入口问题：这条消息应该由通用、技术还是账单流程处理。
转人工和需要澄清是可组合的决策字段，避免把业务领域、会话动作和具体
业务任务放进同一个互斥标签集合。
"""
import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from anthropic import AsyncAnthropic

from core.llm_utils import extract_text_content

logger = logging.getLogger(__name__)


class RouteCategory(str, Enum):
    """入口业务路由；细节由对应 Agent 结合消息和实体继续处理。"""

    GENERAL = "general"
    TECHNICAL = "technical"
    BILLING = "billing"


class UrgencyLevel(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    """路由结果。保留类名以避免调用方把识别层与 HTTP 术语耦合。"""

    route: RouteCategory
    confidence: float
    urgency: UrgencyLevel
    handoff_requested: bool
    needs_clarification: bool
    entities: Dict[str, List[str]]
    reasoning: str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)


# 模板的职责是描述三个 Agent 的稳定责任边界，而不是细分业务任务。
_TEMPLATES: Dict[RouteCategory, List[str]] = {
    RouteCategory.GENERAL: [
        "我的订单现在到哪一步了？",
        "快递一直没有更新怎么办？",
        "怎样修改账户邮箱？",
        "你们周末提供在线服务吗？",
    ],
    RouteCategory.TECHNICAL: [
        "应用启动后一直闪退",
        "登录时提示 401 错误",
        "页面显示 500，无法继续操作",
        "验证码一直收不到",
    ],
    RouteCategory.BILLING: [
        "为什么这笔订单被重复扣款？",
        "退款通常什么时候到账？",
        "电子发票在哪里下载？",
        "订阅扣费金额不对",
    ],
}

_HANDOFF_PATTERNS = (
    "转人工", "人工客服", "人工坐席", "找人工", "真人客服", "人工处理",
)
_SOCIAL_PATTERNS = ("你好", "您好", "嗨", "hello", "hi", "在吗", "谢谢", "感谢")
_COMPLAINT_PATTERNS = ("太差", "糟糕", "不满意", "一直没人", "等了很久", "投诉")
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ("紧急", "emergency", "urgent", "asap", "立刻"),
    UrgencyLevel.HIGH: ("今天", "马上", "尽快", "hurry", "now"),
    UrgencyLevel.MEDIUM: ("这周", "soon", "快点"),
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """使用 LLM、Embedding 和模式规则融合决定三路业务路由。"""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_timeout_s: float = 20.0,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = AsyncAnthropic(**kwargs)
        self.model = model
        self.threshold = confidence_threshold
        self._embedding_api_key = (embedding_api_key or "").strip()
        self._embedding_base_url = (embedding_base_url or "").strip().rstrip("/")
        self._embedding_endpoint = (
            self._embedding_base_url
            if self._embedding_base_url.endswith("/embeddings")
            else f"{self._embedding_base_url}/embeddings"
        )
        self._embedding_model = (embedding_model or "").strip()
        self._embedding_timeout_s = embedding_timeout_s
        self._remote_embedding_enabled = bool(
            self._embedding_api_key and self._embedding_base_url and self._embedding_model
        )
        self._embedding_enabled = True
        self._tpl_embeddings: Dict[RouteCategory, List[List[float]]] = {}
        self._cache: Dict[str, IntentResult] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
        sources: Optional[List[str]] = None,
    ) -> IntentResult:
        active_sources = set(sources or ["llm", "embedding", "pattern"])
        invalid_sources = active_sources - {"llm", "embedding", "pattern"}
        if not active_sources or invalid_sources:
            raise ValueError(f"无效的路由识别来源: {sorted(invalid_sources) or '空列表'}")

        key = self._cache_key(message, history, active_sources)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1
        t0 = time.monotonic()

        llm_task = asyncio.create_task(self._llm_recognize(message, history)) if "llm" in active_sources else None
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if "embedding" in active_sources else None
        pattern = self._pattern_recognize(message) if "pattern" in active_sources else self._empty_result()
        pending = [task for task in (llm_task, emb_task) if task is not None]
        if pending:
            await asyncio.gather(*pending)
        llm = llm_task.result() if llm_task else self._empty_result(failed=True)
        embedding = emb_task.result() if emb_task else self._empty_result()

        route, confidence, source_scores = self._vote(llm, embedding, pattern, active_sources)
        normalized = self._clean_text(message)
        handoff_requested = self._handoff_requested(normalized)
        result = IntentResult(
            route=route,
            confidence=confidence,
            urgency=self._urgency(normalized, handoff_requested),
            handoff_requested=handoff_requested,
            needs_clarification=(confidence < self.threshold and not self._is_social_message(normalized)),
            entities=self._extract_entities(normalized),
            reasoning=str(llm.get("reasoning", "")),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
        )
        if len(self._cache) >= 1000:
            for old_key in list(self._cache)[:500]:
                del self._cache[old_key]
        self._cache[key] = result
        return result

    def learn(self, message: str, correct: RouteCategory) -> None:
        """将人工确认的路由样本加入对应模板，并清除派生缓存。"""
        templates = _TEMPLATES.setdefault(correct, [])
        if message not in templates:
            templates.append(message)
            self._tpl_embeddings.pop(correct, None)
            self._cache.clear()
            logger.info("学习新路由样本 → %s", correct.value)

    async def _llm_recognize(
        self, message: str, history: Optional[List[Dict[str, str]]]
    ) -> Dict[str, Any]:
        examples = "\n".join(
            f'  消息: "{templates[0]}" → 路由: {route.value}'
            for route, templates in _TEMPLATES.items()
        )
        context = ""
        if history:
            context = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(item.get('role', 'user'))}: {self._clean_text(item.get('content', ''))}"
                for item in history[-3:]
            )
        prompt = self._clean_text(f'''你是客服入口路由器。只判断用户问题应交由哪个业务 Agent 处理。
路由定义：general 处理订单、物流、账户、常规咨询、投诉与反馈；technical 处理登录、报错、崩溃、功能异常和配置问题；billing 处理扣款、支付、退款、发票和订阅金额问题。
“转人工”等会话动作由系统独立处理，不能作为路由类别。不要把退款、发票、登录故障等再拆成细粒度标签。

示例:
{examples}
{context}
用户消息: "{self._clean_text(message)}"

只返回 JSON：{{"route":"general|technical|billing", "confidence":0到1, "reasoning":"一句话"}}''')
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=192,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = extract_text_content(response.content)
            start, end = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[start:end])
            data["route"] = RouteCategory(str(data.get("route", "general")))
            data["confidence"] = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
            return data
        except Exception as ex:
            logger.warning("LLM 路由识别失败: %s", ex)
            return self._empty_result(reasoning="LLM 失败", failed=True)

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        try:
            await self._load_template_embeddings()
            message_vector = await self._embed_text(message)
            best_route, best_score = RouteCategory.GENERAL, 0.0
            for route, vectors in self._tpl_embeddings.items():
                score = max(_cosine(message_vector, vector) for vector in vectors)
                if score > best_score:
                    best_route, best_score = route, score
            return {"route": best_route, "confidence": best_score}
        except Exception as ex:
            logger.warning("Embedding 路由识别失败: %s", ex)
            return self._empty_result()

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        patterns = {
            RouteCategory.BILLING: (
                "退款", "退货", "重复扣款", "多扣", "扣费", "支付失败", "支付", "发票", "税号", "订阅", "invoice", "refund",
            ),
            RouteCategory.TECHNICAL: (
                "无法登录", "登录失败", "验证码", "崩溃", "闪退", "报错", "错误码", "页面打不开", "异常", "配置", "error", "crash",
            ),
            RouteCategory.GENERAL: (
                "订单", "物流", "快递", "配送", "账户", "邮箱", "服务", "投诉", "怎么", "什么", "你好", "hello",
            ),
        }
        route, score = self._best_pattern_match(message.lower(), patterns)
        return {"route": route, "confidence": score}

    def _vote(
        self,
        llm: Dict[str, Any],
        embedding: Dict[str, Any],
        pattern: Dict[str, Any],
        active_sources: set[str],
    ) -> tuple[RouteCategory, float, Dict[str, float]]:
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(embedding.get("confidence", 0.0) or 0.0),
            "pattern": float(pattern.get("confidence", 0.0) or 0.0),
        }
        if "llm" in active_sources and llm.get("failed"):
            for source, result in (("embedding", embedding), ("pattern", pattern)):
                if source in active_sources and result.get("confidence", 0.0) > 0:
                    return result["route"], source_scores[source], source_scores
            return RouteCategory.GENERAL, 0.0, source_scores

        base_weights = {"llm": 0.7, "embedding": 0.2, "pattern": 0.1}
        total_weight = sum(base_weights[source] for source in active_sources)
        scores: Dict[RouteCategory, float] = {}
        for source, result in (("llm", llm), ("embedding", embedding), ("pattern", pattern)):
            if source not in active_sources:
                continue
            route = result.get("route", RouteCategory.GENERAL)
            scores[route] = scores.get(route, 0.0) + (base_weights[source] / total_weight) * float(result.get("confidence", 0.0) or 0.0)
        if not scores:
            return RouteCategory.GENERAL, 0.0, source_scores
        route = max(scores, key=scores.get)
        return route, scores[route], source_scores

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        return {
            "order_id": self._unique(re.findall(r"(?:订单号?|order(?:_id)?|#)\s*[:：#]?\s*([A-Za-z0-9_-]{4,32})", message, re.I)),
            "product": [],
            "date": self._unique(re.findall(r"(今天|明天|昨天|本周|这周|下周|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message)),
            "amount": self._unique(re.findall(r"((?:¥|￥)\s*\d+(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?\s*(?:元|块|rmb|cny|usd|美元))", message, re.I)),
            "error_code": self._unique(
                re.findall(r"(?:error(?:_code)?|错误码|状态码|http)\s*[:：#]?\s*([45]\d{2})\b", message, re.I)
                + re.findall(r"\b([45]\d{2})\b", message)
            ),
        }

    async def _load_template_embeddings(self) -> None:
        missing = [route for route in _TEMPLATES if route not in self._tpl_embeddings]
        if not missing:
            return
        texts = [text for route in missing for text in _TEMPLATES[route]]
        vectors = await self._embed_texts(texts)
        index = 0
        for route in missing:
            count = len(_TEMPLATES[route])
            self._tpl_embeddings[route] = vectors[index:index + count]
            index += count

    async def _embed_text(self, text: str) -> List[float]:
        return (await self._embed_texts([text]))[0]

    async def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self._remote_embedding_enabled:
            try:
                headers = {"Authorization": f"Bearer {self._embedding_api_key}", "Content-Type": "application/json"}
                async with httpx.AsyncClient(timeout=self._embedding_timeout_s) as client:
                    response = await client.post(
                        self._embedding_endpoint,
                        headers=headers,
                        json={"model": self._embedding_model, "input": texts},
                    )
                    response.raise_for_status()
                    data = response.json().get("data", [])
                vectors = [list(item["embedding"]) for item in sorted(data, key=lambda item: int(item.get("index", 0)))]
                if len(vectors) != len(texts) or any(not vector for vector in vectors):
                    raise ValueError("Embedding 返回的向量数量或内容不正确")
                return vectors
            except Exception as ex:
                raise RuntimeError(f"远端 Embedding 调用失败: {ex}") from ex
        return [self._local_embedding(text) for text in texts]

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        normalized = text.lower().strip()
        vector = [0.0] * dims
        tokens = {normalized[index:index + size] for size in (1, 2, 3) for index in range(max(0, len(normalized) - size + 1))}
        if not tokens:
            tokens = {normalized}
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            vector[int.from_bytes(digest[:4], "big") % dims] += 1.0 if digest[4] % 2 == 0 else -1.0
        return vector

    @staticmethod
    def _empty_result(reasoning: str = "", failed: bool = False) -> Dict[str, Any]:
        return {"route": RouteCategory.GENERAL, "confidence": 0.0, "reasoning": reasoning, "failed": failed}

    @staticmethod
    def _handoff_requested(message: str) -> bool:
        lowered = message.lower()
        return any(pattern in lowered for pattern in _HANDOFF_PATTERNS)

    @staticmethod
    def _is_social_message(message: str) -> bool:
        normalized = message.lower().strip(" !！。,.？?")
        return len(normalized) <= 12 and any(pattern in normalized for pattern in _SOCIAL_PATTERNS)

    def _urgency(self, message: str, handoff_requested: bool) -> UrgencyLevel:
        lowered = message.lower()
        for level, keywords in _URGENCY_KEYWORDS.items():
            if any(keyword in lowered for keyword in keywords):
                return level
        if handoff_requested:
            return UrgencyLevel.HIGH
        if any(pattern in lowered for pattern in _COMPLAINT_PATTERNS):
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None, sources: Optional[set[str]] = None) -> str:
        payload: Dict[str, Any] = {"message": self._clean_text(message)[:200], "sources": sorted(sources or {"llm", "embedding", "pattern"})}
        if history:
            payload["history"] = [{"role": self._clean_text(item.get("role", ""))[:20], "content": self._clean_text(item.get("content", ""))[:160]} for item in history[-3:]]
        return hashlib.md5(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _best_pattern_match(message: str, patterns: Dict[RouteCategory, tuple[str, ...]]) -> tuple[RouteCategory, float]:
        best_route, best_score = RouteCategory.GENERAL, 0.0
        for route, keywords in patterns.items():
            hits = sum(keyword in message for keyword in keywords)
            if hits:
                score = min(1.0, 0.5 + 0.25 * (hits - 1))
                if score > best_score:
                    best_route, best_score = route, score
        return best_route, best_score

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _clean_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {"size": len(self._cache), "hits": self.cache_hits, "misses": self.cache_misses, "hit_rate": self.cache_hits / total if total else 0.0}
