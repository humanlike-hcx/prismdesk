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
    assert "billing" in prompt
    assert '"route":"general|technical|billing"' in prompt
    assert result["route"] == RouteCategory.BILLING


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


def test_pattern_only_mode_returns_route_and_independent_handoff_flag():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    recognizer.threshold = 0.5
    recognizer._cache = {}
    recognizer.cache_hits = 0
    recognizer.cache_misses = 0

    result = asyncio.run(recognizer.recognize("转人工处理电子发票", sources=["pattern"]))

    assert result.route == RouteCategory.BILLING
    assert result.handoff_requested is True
    assert result.source_scores["pattern"] == 0.5


def test_source_mode_is_part_of_cache_key():
    recognizer = IntentRecognizer.__new__(IntentRecognizer)
    pattern_key = recognizer._cache_key("退款多久到账", sources={"pattern"})
    fusion_key = recognizer._cache_key("退款多久到账", sources={"llm", "embedding", "pattern"})

    assert pattern_key != fusion_key
