import unittest

from semantic_search.semantic_search_driver import Retriever, ReRanker


class TestReranker(unittest.TestCase):
    def test_rerank(self):
        docs = ["dogs are friendly to human", "I like eating fish",
                "Snorkeling is wonderful as you can see wonderful colorful fishes "]
        retriever = Retriever(backend="faiss")

        r = retriever.initialize_backend()
        assert isinstance(r, dict)
        assert r["status"] == "OK"

        r = retriever.build_index(documents=docs)
        assert isinstance(r, dict)
        assert r["status"] == "OK"
        q = "what can I enjoy if I go to the see ?"
        k = 2
        retriever_results = retriever.search(query=q, k=k)
        assert isinstance(retriever_results, dict)
        assert isinstance(retriever_results["indices"], list)
        assert len(retriever_results["indices"]) == k
        assert set(retriever_results["indices"]) == {1, 2}
        short_list = [docs[i] for i in retriever_results["indices"]]
        reranker = ReRanker()
        results = reranker.rerank(query=q, k=k, documents=short_list)
        assert isinstance(results, dict)
        assert results["status"] == "OK"
        assert results["indices"][0]==0

