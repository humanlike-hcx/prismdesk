import asyncio

from core.intent_recognizer import IntentRecognizer, RouteCategory


class FakeMessages:
    def __init__(self, content):
        self.content = content
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        block = type("TextBlock", (), {"type": "text", "text": self.content})()
        return type("Response", (), {"content": [block]})()


def test_few_shot_examples_describe_three_routes():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer.model = "test-model"
    messages = FakeMessages('{"route":"billing","confidence":0.9,"reasoning":"退款问题"}')
    recognizer.client = type("Client", (), {"messages": messages})()

    result = asyncio.run(recognizer._llm_recognize("这笔钱什么时候退回来", None))

    prompt = messages.calls[0]["messages"][0]["content"]
    assert '消息: "付款成功但订单没有生效" → 路由: billing' in prompt
    assert '消息: "输入正确密码后仍无法登录" → 路由: technical' in prompt
    assert '"route":"general|technical|billing"' in prompt
    assert result["route"] == RouteCategory.BILLING
    assert messages.calls[0]["temperature"] == 0.0


def test_llm_result_requires_a_json_object():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer.model = "test-model"
    messages = FakeMessages("billing")
    recognizer.client = type("Client", (), {"messages": messages})()

    result = asyncio.run(recognizer._llm_recognize("退款多久到账", None))

    assert result["failed"] is True
    assert len(messages.calls) == 2


def test_remote_embedding_batches_inputs_and_restores_response_order(monkeypatch):
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._remote_embedding_enabled = True
    recognizer._embedding_endpoint = "https://embedding.example/v1/embeddings"
    recognizer._embedding_api_key = "secret"
    recognizer._embedding_model = "test-embedding"
    recognizer._embedding_timeout_s = 5
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}

    class Client:
        def __init__(self, **kwargs):
            captured["client"] = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            captured["url"] = url
            captured["request"] = kwargs
            return Response()

    monkeypatch.setattr("core.intent_recognizer.httpx.AsyncClient", Client)
    vectors = asyncio.run(recognizer._embed_texts(["退款", "登录失败"]))

    assert captured["url"] == "https://embedding.example/v1/embeddings"
    assert captured["request"]["json"]["input"] == ["退款", "登录失败"]
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]


def test_handoff_and_urgency_rules_are_independent_from_route_sources():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    assert recognizer._handoff_requested("请转人工处理电子发票") is True
    assert recognizer._handoff_requested("电子发票在哪里下载") is False


def test_invalid_route_source_is_rejected():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer._cache = {}
    recognizer.cache_hits = 0
    recognizer.cache_misses = 0
    try:
        asyncio.run(recognizer.recognize("退款多久到账", sources=["pattern"]))
    except ValueError as exc:
        assert "无效的路由识别来源" in str(exc)
    else:
        raise AssertionError("pattern must not participate in business routing")


def test_source_mode_is_part_of_cache_key():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    embedding_key = recognizer._cache_key("退款多久到账", sources={"embedding"})
    fusion_key = recognizer._cache_key("退款多久到账", sources={"llm", "embedding"})

    assert embedding_key != fusion_key


def test_two_source_vote_reports_margin_for_clarification_decision():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    route, confidence, margin, source_scores = recognizer._vote(
        {"route": RouteCategory.TECHNICAL, "confidence": 0.8},
        {"route": RouteCategory.BILLING, "confidence": 0.76},
        {"llm", "embedding"},
    )

    assert route == RouteCategory.TECHNICAL
    assert confidence == 0.56
    assert margin == 0.332
    assert source_scores == {"llm": 0.8, "embedding": 0.76}
