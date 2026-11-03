import pytest

from sec_rag.indexing import embedder as embedder_module
from sec_rag.indexing.embedder import ATLAS_API_URL, VOYAGE_API_URL, Embedder
from sec_rag.indexing.vector_store import VectorStore


class FakeResponse:
    def __init__(self, status_code, body=None, headers=None):
        self.status_code = status_code
        self.ok = status_code < 400
        self._body = body or {}
        self.headers = headers or {}
        self.text = str(body)

    def json(self):
        return self._body


class FakeSession:
    """Records requests; answers with queued failures first, then embeddings."""

    def __init__(self, failures=()):
        self.failures = list(failures)
        self.requests = []

    def post(self, url, json, headers, timeout):
        self.url = url
        self.requests.append(json)
        if self.failures:
            return self.failures.pop(0)
        # Return items out of order, as the API may; each vector encodes its text's index.
        data = [{"index": i, "embedding": [float(i)]} for i in range(len(json["input"]))]
        return FakeResponse(200, {"data": data[::-1], "usage": {"total_tokens": 7}})


def make_embedder(session, **kwargs):
    return Embedder(model_name="voyage-finance-2", api_key="test-key", session=session, **kwargs)


def test_documents_are_batched_in_order():
    session = FakeSession()
    vectors = make_embedder(session, batch_size=2).embed_documents(["a", "b", "c"])

    assert [len(r["input"]) for r in session.requests] == [2, 1]
    assert all(r["input_type"] == "document" for r in session.requests)
    assert vectors == [[0.0], [1.0], [0.0]]


def test_query_uses_query_input_type():
    session = FakeSession()
    embedder = make_embedder(session)
    assert embedder.embed_query("revenue?") == [0.0]
    assert session.requests[0] == {
        "input": ["revenue?"], "model": "voyage-finance-2", "input_type": "query"
    }
    assert embedder.total_tokens == 7


def test_rate_limit_is_retried(monkeypatch):
    sleeps = []
    monkeypatch.setattr(embedder_module.time, "sleep", sleeps.append)
    session = FakeSession(failures=[FakeResponse(429, headers={"Retry-After": "3"})])

    assert make_embedder(session).embed_query("q") == [0.0]
    assert sleeps == [3.0]
    assert len(session.requests) == 2


def test_client_errors_are_not_retried():
    session = FakeSession(failures=[FakeResponse(401, {"detail": "bad key"})])
    with pytest.raises(RuntimeError, match="HTTP 401"):
        make_embedder(session).embed_query("q")
    assert len(session.requests) == 1


def test_missing_api_key_fails_clearly():
    embedder = Embedder(api_key=None, session=FakeSession())
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        embedder.embed_query("q")


def test_store_from_another_model_is_stale_until_reset(tmp_path):
    old = VectorStore(path=tmp_path, collection_name="chunks", embedding_model="old-model")
    old.collection.add(ids=["1"], embeddings=[[1.0, 0.0]], documents=["x"])

    store = VectorStore(path=tmp_path, collection_name="chunks", embedding_model="new-model")
    assert store.built_with == "old-model"
    assert store.is_stale
    with pytest.raises(RuntimeError, match="sec-rag index"):
        store.query([1.0, 0.0], k=1)

    store.reset()
    assert store.built_with == "new-model"
    assert not store.is_stale
    assert store.count() == 0


@pytest.mark.parametrize(
    ("key", "url"), [("al-abc", ATLAS_API_URL), ("pa-abc", VOYAGE_API_URL)]
)
def test_endpoint_follows_the_key(key, url):
    session = FakeSession()
    Embedder(api_key=key, api_url=None, session=session).embed_query("q")
    assert session.url == url
