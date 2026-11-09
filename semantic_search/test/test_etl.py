import unittest
from typing import Iterable
from semantic_search.semantic_search_driver import ETL


class TestETL(unittest.TestCase):
    def setUp(self) -> None:
        self.etl = ETL(dataset_name="scifact")

    def test_load_sci_fact(self):
        docs: Iterable[str] = self.etl.load("train")
        assert docs is not None
        assert len(docs) > 0
        k: int = 10
        for i in range(k):
            assert isinstance(docs[i], str)
            assert len(docs[i]) > 0


if __name__ == "__main__":
    unittest.main()
