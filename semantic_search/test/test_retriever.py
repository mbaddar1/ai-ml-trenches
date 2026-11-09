import unittest

from semantic_search.semantic_search_driver import Retriever


class TestRetriever(unittest.TestCase):
    def setUp(self):
        pass

    def test_initialize_lexical_backend(self):
        retriever = Retriever(backend="bm25")
        r = retriever.initialize_backend()
        assert isinstance(r, dict)
        assert r["status"] == "OK"

    def test_initialize_vector_index(self):
        retriever = Retriever(backend="faiss")
        r = retriever.initialize_backend()
        assert isinstance(r, dict)
        assert r["status"] == "OK"

    def test_build_lexical_index(self):
        retriever = Retriever(backend="bm25")
        r = retriever.initialize_backend()
        assert isinstance(r, dict)
        assert r["status"] == "OK"

        docs = ["dogs are really friendly to human","I like eating fish"]
        r = retriever.build_index(docs)
        assert isinstance(r, dict)
        assert r["status"] == "OK"
    def test_build_vector_index(self):
        retriever = Retriever(backend="faiss")
        r = retriever.initialize_backend()
        assert isinstance(r, dict)
        assert r["status"] == "OK"

        docs = ["dogs are really friendly to human", "I like eating fish"]
        r = retriever.build_index(docs)
        assert isinstance(r, dict)
        assert r["status"] == "OK"

    def test_search_lexical(self):
        retriever = Retriever(backend="bm25")
        r = retriever.initialize_backend()
        assert isinstance(r, dict)
        assert r["status"] == "OK"

        docs = ["dogs are really friendly to human", "I like eating fish"]
        r = retriever.build_index(docs)
        assert isinstance(r, dict)
        assert r["status"] == "OK"
        query = "dogs are nice"
        results = retriever.search(query=query, k=1)
        assert isinstance(results, dict)
        assert "status" in results.keys()
        assert results["status"] == "OK"
        assert results["indices"][0] == 0

    def test_search_vector(self):
        retriever = Retriever(backend="faiss")
        r = retriever.initialize_backend()
        assert isinstance(r, dict)
        assert r["status"] == "OK"

        docs = ["dogs are really friendly to human", "I like eating fish"]
        r = retriever.build_index(docs)
        assert isinstance(r, dict)
        assert r["status"] == "OK"
        query = "dogs are nice"
        results = retriever.search(query=query, k=1)
        assert isinstance(results, dict)
        assert "status" in results.keys()
        assert results["status"] == "OK"
        assert results["indices"][0] == 0





if __name__ == '__main__':
    unittest.main()
