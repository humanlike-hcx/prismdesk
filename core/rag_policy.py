"""独立于业务路由的 RAG 调用策略。"""
from enum import Enum
from typing import Any


class RAGPolicy(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    FORBIDDEN = "forbidden"


_SOCIAL_PATTERNS = ("你好", "您好", "嗨", "hello", "hi", "谢谢", "感谢")


def decide_rag_policy(
    message: str,
    route: Any = None,
    *,
    handoff_requested: bool = False,
) -> RAGPolicy:
    """根据业务域和会话动作决定是否在生成前检索知识库。

    技术和账单回复依赖产品规则、故障说明或售后政策，因此预取知识；人工
    交接不检索，避免把额外答案混入交接流程。通用域由 Agent 按上下文选择。
    """
    text = (message or "").strip().lower()
    route_value = getattr(route, "value", route)
    if handoff_requested:
        return RAGPolicy.FORBIDDEN
    if route_value in {"technical", "billing"}:
        return RAGPolicy.REQUIRED
    normalized = text.strip(" !！。,.？?")
    if len(normalized) <= 12 and any(pattern in normalized for pattern in _SOCIAL_PATTERNS):
        return RAGPolicy.FORBIDDEN
    return RAGPolicy.OPTIONAL
