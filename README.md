# PrismDesk

基于 FastAPI 的智能客服后端，提供多 Agent 路由、知识库检索、会话记忆、业务规则加载、监控和评测接口。

## 启动

需要 Docker Compose 和可用的模型服务。复制配置模板：

```bash
cp .env.example .env
```

填写 `ANTHROPIC_API_KEY`、`ANTHROPIC_MODEL`；兼容服务另填 `ANTHROPIC_BASE_URL`。设置 `REDIS_PASSWORD` 后启动：

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
