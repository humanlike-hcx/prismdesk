# PrismDesk

基于 FastAPI 的智能客服后端，提供多 Agent 路由、知识库检索、会话记忆、业务规则加载、监控和评测接口。

## 启动

需要 Docker Compose 和可用的模型服务。复制配置模板：

```bash
cp .env.example .env
```

填写 `ANTHROPIC_API_KEY`、`ANTHROPIC_MODEL`；兼容服务另填 `ANTHROPIC_BASE_URL`。设置 `REDIS_PASSWORD` 后启动：

意图识别如需使用远程 Embedding，在 `.env` 中填写 OpenAI 兼容接口配置：

```env
EMBEDDING_API_KEY=你的密钥
EMBEDDING_BASE_URL=https://服务商地址/v1
EMBEDDING_MODEL=服务商提供的模型名
EMBEDDING_TIMEOUT_S=20
```

`EMBEDDING_BASE_URL` 可填到 `/v1`，程序会请求其 `/embeddings` 端点；也可直接填写完整的 `/v1/embeddings` 地址。未配置这三项时使用本地字符 n-gram 向量。

```bash
docker compose up -d --build
docker compose logs -f prismdesk
```

- API 文档：[localhost:8000/docs](http://localhost:8000/docs)
- 健康检查：[localhost:8000/health](http://localhost:8000/health)

## 目录

| 目录 | 用途 |
| --- | --- |
| `api/` | HTTP 接口与服务入口 |
| `agents/` | Agent 编排与工具执行 |
| `core/` | 意图识别、模型工具与规则加载 |
| `mcp/` | 知识库与检索工具 |
| `memory/` | Redis、ChromaDB 会话记忆 |
| `skills/` | 运行时业务规则 |
| `monitor/`、`evaluation/` | 监控与评测 |
| `config/` | Nginx、Prometheus 配置 |
| `tests/` | 自动化测试 |

本地密钥保存在 `.env`，运行数据写入 `data/`、`logs/`，均不纳入版本控制。知识库初始为空，可通过 `/knowledge/add` 或 `/knowledge/upload` 导入业务资料。

`/chat` 返回三路业务 `route`（`general`、`technical`、`billing`）以及独立的
`handoff_requested`、`needs_clarification` 和 `rag_policy`（`required`、`optional`、`forbidden`）。技术与账单问题会先检索知识库；明确人工交接和简单寒暄不会检索，避免无关知识干扰交接或问候。

路由评测默认读取 `evaluation/datasets/route_validation_v1.jsonl`：三类各 12 条，且不复用 Embedding 模板或 LLM Few-shot。调用 `/eval/run` 时可用 `route_suite` 指定 `validation`、`dev` 或 `blind`，并用 `route_sources: ["embedding"]`、`route_sources: ["llm"]` 或 `route_sources: ["llm", "embedding"]` 做消融。`blind` 是参数冻结后的项目内留出集，不能替代独立业务盲测；详情见 `evaluation/README.md`。
