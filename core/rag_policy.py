"""RAG 调用策略。

策略由意图识别结果决定，避免把“是否检索”完全交给生成模型临场判断。
"""
from enum import Enum
from typing import Any


class RAGPolicy(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    FORBIDDEN = "forbidden"


_REQUIRED_INTENTS = {
    "query",
    "technical",
    "billing",
    "complaint",
    "order_status",
    "logistics",
    "refund",
    "invoice",
    "payment_issue",
    "account_security",
    "technical_login",
    "technical_crash",
}
_FORBIDDEN_INTENTS = {"greeting", "feedback", "escalation", "human_handoff"}
_BUSINESS_KEYWORDS = {
    "退款", "退货", "订单", "物流", "配送", "发票", "扣款", "支付", "账单",
    "订阅", "登录", "报错", "错误", "崩溃", "会员", "积分", "账户", "密码",
    "地址", "refund", "order", "invoice", "payment", "error", "login",
}


def decide_rag_policy(message: str, intent: Any = None) -> RAGPolicy:
    """根据细粒度意图和业务关键词返回稳定、可测试的检索策略。"""
    text = (message or "").strip().lower()
    intent_value = getattr(intent, "value", intent)
    if intent_value in {"escalation", "human_handoff"}:
        return RAGPolicy.FORBIDDEN
    if intent_value in _REQUIRED_INTENTS:
        return RAGPolicy.REQUIRED
    if any(keyword in text for keyword in _BUSINESS_KEYWORDS):
        return RAGPolicy.REQUIRED
    if intent_value in _FORBIDDEN_INTENTS:
        return RAGPolicy.FORBIDDEN
    return RAGPolicy.OPTIONAL
