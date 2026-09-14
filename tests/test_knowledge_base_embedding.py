from mcp.knowledge_base import KnowledgeBase


class FakeCollection:
    def __init__(self):
        self.upsert_kwargs = None
        self.query_kwargs = None

    def count(self):
        return 1

    def upsert(self, **kwargs):
        self.upsert_kwargs = kwargs

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "documents": [["退款标准"]],
            "metadatas": [[{"title": "KB-101", "chunk_index": 0}]],
            "distances": [[0.1]],
        }


def test_remote_embedding_is_used_for_both_indexing_and_query():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._remote_embedding_enabled = True
    kb._collection = FakeCollection()
    kb._embed_texts = lambda texts: [[float(index), 1.0] for index, _ in enumerate(texts)]

    assert kb.add_documents([{"title": "KB-101", "content": "退款标准"}]) == 1
    assert kb._collection.upsert_kwargs["embeddings"] == [[0.0, 1.0]]

    results = kb.search("怎么退款", top_k=1)
    assert "query_embeddings" in kb._collection.query_kwargs
    assert "query_texts" not in kb._collection.query_kwargs
    assert results[0]["score"] == 0.9
