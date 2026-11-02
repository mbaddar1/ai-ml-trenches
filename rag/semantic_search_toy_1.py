"""
Semantic Search Coding Exercise 1
"""
import sqlite3
import Stemmer
import numpy as np
import pandas as pd
import bm25s
from loguru import logger
from sentence_transformers import SentenceTransformer


class DataProcessor:
    def __init__(self):
        self.conn = sqlite3.connect("restaurants.db")

    def load_db(self):
        # load csv files
        restaurants_df = pd.read_csv("data/restaurants.csv", sep=",")
        reviews_df = pd.read_csv("data/reviews.csv", sep=",")
        # add to db
        restaurants_df.to_sql("restaurants", self.conn, if_exists="replace", index=False)
        reviews_df.to_sql("reviews", self.conn, if_exists="replace", index=False)
        logger.info(f"Data loaded successfully")

    def process_data(self):
        query = """
            SELECT
                r.restaurant_id,
                r.name,
                GROUP_CONCAT(rv.review_text, ' ') AS review_text_concat
            FROM restaurants r
            INNER JOIN reviews rv
                ON r.restaurant_id = rv.restaurant_id
            GROUP BY r.restaurant_id, r.name;
        """
        result = pd.read_sql_query(query, self.conn)
        logger.info("Data preprocessed successfully")
        return result

    def run(self):
        self.load_db()
        df = self.process_data()
        assert df.shape[0] > 0 and df.shape[1] >= 0
        return df


class Retriever:
    def __init__(self):
        self.retriever = bm25s.BM25()
        self.stemmer = Stemmer.Stemmer("english")
        self.index_built = False
        self.corpus = None

    def build_index(self, df: pd.DataFrame):
        col_name = "review_text_concat"
        assert self.corpus is None, "Corpus already built"
        self.corpus = df[col_name].to_list()
        self.stemmer = Stemmer.Stemmer("english")
        # Tokenize the corpus and only keep the ids (faster and saves memory)
        corpus_tokens = bm25s.tokenize(self.corpus, stopwords="en", stemmer=self.stemmer)
        # Create the BM25 model and index the corpus
        self.retriever.index(corpus_tokens)
        self.index_built = True
        logger.info(f"Index built successfully")

    def retrieve(self, query: str, k: int):
        query_tokens = bm25s.tokenize(query, stopwords="en", stemmer=self.stemmer)
        results = self.retriever.retrieve(query_tokens=query_tokens, k=k)
        doc_idx = list(results.documents.flatten())
        documents = [self.corpus[i] for i in doc_idx]
        return documents, results


class SemanticReRanker:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info(f"Semantic Ranker model initialized successfully")

    def rerank(self, query, documents):
        query_embedding = self.model.encode([query], show_progress_bar=True)
        doc_embeddings = self.model.encode(documents, show_progress_bar=True)
        scores = query_embedding @ doc_embeddings.T
        sorted_idx = np.argsort(scores)
        return sorted_idx


if __name__ == "__main__":
    # Get processed data
    data_processor = DataProcessor()
    processed_df = data_processor.run()

    # indexing phase
    retriever = Retriever()
    retriever.build_index(processed_df)

    # retrieval
    q = "Pizza with outdoor seating"
    documents, meta_data = retriever.retrieve(query=q, k=3)
    logger.info(f"Retrieval results ( n = {len(documents)} ): \n{documents}")

    # reranking
    reranker = SemanticReRanker()
    sorted_idx = reranker.rerank(query=q, documents=documents)
    sorted_documents = [documents[i] for i in list(sorted_idx.flatten())]
    logger.info(f"ReRanked docs = \n {sorted_documents}")
