from core.intent_recognizer import RouteCategory
from core.rag_policy import RAGPolicy, decide_rag_policy


def test_technical_and_billing_routes_require_retrieval():
    assert decide_rag_policy("退款什么时候到账", RouteCategory.BILLING) == RAGPolicy.REQUIRED
    assert decide_rag_policy("页面提示 500", RouteCategory.TECHNICAL) == RAGPolicy.REQUIRED


def test_handoff_and_social_messages_forbid_retrieval():
    assert decide_rag_policy("转人工处理退款", RouteCategory.BILLING, handoff_requested=True) == RAGPolicy.FORBIDDEN
    assert decide_rag_policy("你好", RouteCategory.GENERAL) == RAGPolicy.FORBIDDEN


def test_general_route_leaves_retrieval_to_agent_context():
    assert decide_rag_policy("这笔订单怎么还没到", RouteCategory.GENERAL) == RAGPolicy.OPTIONAL
    assert decide_rag_policy("我想了解一下", RouteCategory.GENERAL) == RAGPolicy.OPTIONAL
