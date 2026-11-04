import pytest

from sec_rag.indexing.vector_store import VectorStore


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
