from core.intent_recognizer import IntentCategory
from core.rag_policy import RAGPolicy, decide_rag_policy


def test_business_intents_require_retrieval():
    assert decide_rag_policy("退款什么时候到账", IntentCategory.REFUND) == RAGPolicy.REQUIRED
    assert decide_rag_policy("页面提示 500", IntentCategory.TECHNICAL_CRASH) == RAGPolicy.REQUIRED


def test_social_and_handoff_intents_forbid_retrieval():
    assert decide_rag_policy("你好", IntentCategory.GREETING) == RAGPolicy.FORBIDDEN
    assert decide_rag_policy("转人工", IntentCategory.HUMAN_HANDOFF) == RAGPolicy.FORBIDDEN


def test_uncertain_business_message_uses_keyword_safety_net():
    assert decide_rag_policy("这笔订单怎么还没到", IntentCategory.OTHER) == RAGPolicy.REQUIRED
    assert decide_rag_policy("你好，我想申请退款", IntentCategory.GREETING) == RAGPolicy.REQUIRED
    assert decide_rag_policy("我想了解一下", IntentCategory.OTHER) == RAGPolicy.OPTIONAL
