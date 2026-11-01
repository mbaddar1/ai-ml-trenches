"""
Based on this dataset
https://zenodo.org/records/18723813

Problem to Solve
-----------------------
We want to design a RAG system over a set of restaurant datasets to answer some questions
1- Give me top 3 restaurants in city x having the cuisine y
2- Give me restaurant in city x with kids area and a parking spot

# TODO
    # iteration 1
    1. Just works
    2. Evaluation
    # Iteration 2
    1. Experiment different embedding models
    2. Experiment different generation models

"""
import os
import pickle
from fileinput import input
from typing import Tuple

import faiss
import numpy as np
import ollama
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch


class RAGEngine:
    def __init__(self, embedding_model_name: str, language_model_name: str, index_backend: str, device: str):
        self.embedding_model = SentenceTransformer(model_name_or_path=embedding_model_name, device=device)
        logger.info(f"loaded embedding model {embedding_model_name} with device = {self.embedding_model.device}")
        self.embedding_dim = self.embedding_model.get_embedding_dimension()
        if index_backend == "faiss":
            self.vector_index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("initialized vector index with index-backend {} embedding_dim = {}".format(index_backend,
                                                                                                   self.embedding_dim))
            logger.info(f"Index is trained ? = {self.vector_index.is_trained}")
        else:
            raise NotImplementedError(f"Index backend {index_backend} not implemented")
        self.index_backend = index_backend
        self.textual_data = None
        self.language_model_name = language_model_name

    def load_or_build_index(self, df: pd.DataFrame, index_path: str):
        if os.path.exists(index_path):
            logger.info(f"Loading index from {index_path}")
            with open(index_path, "rb") as f:
                loaded_data = pickle.load(f)
                self.vector_index = loaded_data["index"]
                self.textual_data = loaded_data["textual_data"]
                f.close()
        else:
            logger.info(f"Couldn't find saved index, building index from df with shape = {df.shape}")
            self.build_index(df=df, index_path=index_path)

    def build_index(self, df: pd.DataFrame, index_path: str):
        # Generate batch of strings
        logger.info(f"building vector index using df with shape = {df.shape}")

        logger.info(f"converting df rows to batch of strings")
        self.textual_data = RAGEngine.__rows_to_strings(df=df)

        logger.info(f"Generating embeddings")
        embeddings = self.embedding_model.encode(inputs=self.textual_data, batch_size=128, show_progress_bar=True)
        logger.info("Successfully generated embeddings matrix of shape {},adding it to the vector index"
                    .format(embeddings.shape))
        # initialize IndexL2Flat # TODO accurate but not very fast, explore alternatives
        if self.index_backend == "faiss":
            self.vector_index.add(embeddings)
            with open(index_path, "wb") as f:
                logger.info(f"saving vector index to {index_path}")
                pickle.dump({"index": self.vector_index, "textual_data": self.textual_data}, f)
                f.close()
            logger.info(f"Added embeddings matrix of shape {embeddings.shape} to index ")
            logger.info(f"Index built successfully with num of total data points = {self.vector_index.ntotal}")
        else:
            raise NotImplementedError(f"Index backend {self.index_backend} not implemented")

    def __search(self, query, k) -> Tuple[np.ndarray, np.ndarray]:
        query_embedding = self.embedding_model.encode([query])
        distances, search_results_index = self.vector_index.search(x=query_embedding, k=k)
        return distances, search_results_index

    def update(self):
        pass

    def answer(self, query: str, k: int):
        distances, search_results_index = self.__search(query, k)
        idx = search_results_index.flatten()
        textual_results = [self.textual_data[i] for i in idx]
        context = "\n\n".join(textual_results)
        logger.info(f"Generated context for the RAG answering step...")
        logger.info(f"Generating answer...")
        stream = ollama.chat(
            model=self.language_model_name,
            messages=[
                {'role': 'system', 'content': context},
                {'role': 'user', 'content': query},
            ],
            stream=True,
        )
        response_chunks = []
        for chunk in stream:
            response_chunks.append(chunk['message']['content'])
        response = ''.join(response_chunks)
        return response

    @staticmethod
    def __rows_to_strings(df, sep=', '):
        result = []
        for _, row in df.iterrows():
            result.append(sep.join(f"{col}: {val}" for col, val in row.items()))
        return result


class RAGEval:
    pass


if __name__ == "__main__":
    # configs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    index_backend = "faiss"
    index_path = "data/index.pkl"
    # load the dataset
    input_df = pd.read_csv("data/yelp_sentiment_master_dataset.csv")
    logger.info(f"loaded data with {input_df.shape[0]} rows and {input_df.shape[1]} columns")
    # data quick understanding
    logger.info("Data columns = {}".format(input_df.columns))
    agg = input_df.groupby('name').agg(n_reviews=('Sl No', 'nunique')).sort_values(by=['n_reviews'], ascending=False)
    logger.info(f"n business = {len(agg)}")
    logger.info(f"agg = \n{agg.head(10)}")
    # build rag engine
    embedding_model_name = "bert-base-nli-mean-tokens"
    language_model = 'hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF'
    rag_engine = RAGEngine(embedding_model_name=embedding_model_name, language_model_name=language_model,
                           index_backend=index_backend, device=device)
    rag_engine.load_or_build_index(df=input_df, index_path=index_path)
    #
    query = "Give me top 3 pizza restaurants in Philadelphia"
    response = rag_engine.answer(query=query, k=20)
    logger.info(f"Query = {query}\n\n Response = {response}")
