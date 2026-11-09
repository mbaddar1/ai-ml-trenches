"""
https://www.datacamp.com/tutorial/claude-code-best-practices

"""
import os

import numpy as np
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from loguru import logger
from typing import Iterable
from typing import List, Optional

class ETL:
    SUPPORTED_DATASETS = ["scifact"]

    def __init__(self, dataset_name: str):
        assert dataset_name in ETL.SUPPORTED_DATASETS
        self.dataset_name = dataset_name

    def load(self, split: str):
        if self.dataset_name == "scifact":
            return self.load_sci_fact(split)
        else:
            raise NotImplementedError("dataset not implemented : {}".format(self.dataset_name))

    def load_sci_fact(self, split: str) -> Iterable[str]:
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        out_dir = os.path.join(os.getcwd(), "datasets")

        # Downloads and unzips the dataset (skips the download if it already exists)
        data_path = util.download_and_unzip(url, out_dir)

        # corpus: abstracts, queries: claims {query_id: claim_text}, qrels: relevance judgments
        _, queries, _ = GenericDataLoader(data_folder=data_path).load(split="test")

        # In BEIR, the SciFact "queries" are the scientific claims
        return list(queries.values())




class Retriever:
    BACK_ENDs = ["bm25", "faiss"]
    DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, backend: str, embedding_model: Optional[str] = None):
        self.backend = backend.lower() if isinstance(backend, str) else backend
        # Only relevant for the vector backend
        self.embedding_model = embedding_model or self.DEFAULT_EMBEDDING_MODEL

        self._bm25 = None          # BM25 module (lexical backend)
        self._bm25_index = None    # BM25 retriever object
        self._model = None         # SentenceTransformer (vector backend)
        self._faiss_index = None   # faiss index
        self._documents: List[str] = []
        self._initialized = False
        self._indexed = False

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _ok(**extra) -> dict:
        return {"status": "OK", **extra}

    @staticmethod
    def _fail(reason: str, **extra) -> dict:
        return {"status": "Failure", "reason": reason, **extra}

    # --------------------------------------------------------------- initialize
    def initialize_backend(self) -> dict:
        """Initialize the BM25 module or the sentence-transformers model."""
        try:
            if self.backend not in self.BACK_ENDs:
                return {"status": "Failure",
                        "Reason": f"Unknown backend '{self.backend}'. Supported: {self.BACK_ENDs}"}

            if self.backend == "bm25":
                import BM25  # pip install BM25
                self._bm25 = BM25
            else:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model)

            self._initialized = True
            self._indexed = False
            return {"status": "OK", "Reason": ""}
        except Exception as e:
            self._initialized = False
            return {"status": "Failure", "Reason": str(e)}

    # -------------------------------------------------------------------- build
    def build_index(self, documents: Iterable[str]) -> dict:
        """Build the BM25 index or the FAISS index over the given documents."""
        try:
            if not self._initialized:
                return {"status": "Failure",
                        "Reason": "Backend not initialized. Call initialize_backend() first."}

            docs = [d for d in documents if isinstance(d, str)]
            if not docs:
                return {"status": "Failure", "Reason": "No documents provided to index."}

            if self.backend == "bm25":
                index = self._bm25.index(docs)
                if index is None:
                    return {"status": "Failure", "Reason": "BM25 indexing returned no index."}
                self._bm25_index = index
            else:
                import faiss

                embeddings = self._model.encode(
                    docs,
                    convert_to_numpy=True,
                    normalize_embeddings=True,   # inner product == cosine similarity
                    show_progress_bar=False,
                ).astype("float32")

                if embeddings.ndim != 2 or embeddings.shape[0] != len(docs):
                    return {"status": "Failure",
                            "Reason": f"Unexpected embedding shape {embeddings.shape}."}

                index = faiss.IndexFlatIP(embeddings.shape[1])
                index.add(embeddings)
                if index.ntotal != len(docs):
                    return {"status": "Failure",
                            "Reason": f"FAISS indexed {index.ntotal} of {len(docs)} documents."}
                self._faiss_index = index

            self._documents = docs
            self._indexed = True
            return {"status": "OK", "Reason": ""}
        except Exception as e:
            self._indexed = False
            return {"status": "Failure", "Reason": str(e)}

    # ------------------------------------------------------------------- search
    def search(self, query: str, k: int) -> dict:
        """Return top-k documents, scores and document indices for a query string."""
        try:
            if not self._indexed:
                return {"status": "Failure", "results": [], "scores": [], "indices": [],
                        "reason": "Index not built. Call build_vector_index() first."}
            if not isinstance(query, str) or not query.strip():
                return {"status": "Failure", "results": [], "scores": [], "indices": [],
                        "reason": "Query must be a non-empty string."}
            if not isinstance(k, int) or k <= 0:
                return {"status": "Failure", "results": [], "scores": [], "indices": [],
                        "reason": "k must be a positive integer."}

            k = min(k, len(self._documents))
            results: List[str] = []
            scores: List[float] = []
            indices: List[int] = []

            if self.backend == "bm25":
                hits = self._bm25_index.search([query], k=k)[0]
                for hit in hits:
                    # Use the index returned by the library if present; otherwise
                    # fall back to locating the document text in the indexed corpus.
                    idx = hit.get("id", hit.get("index"))
                    if idx is None:
                        idx = self._documents.index(hit["document"])
                    results.append(hit["document"])
                    scores.append(float(hit["score"]))
                    indices.append(int(idx))
            else:
                q_emb = self._model.encode(
                    [query],
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).astype("float32")
                dists, ids = self._faiss_index.search(q_emb, k)
                for score, idx in zip(dists[0], ids[0]):
                    if idx == -1:   # FAISS pads with -1 when fewer hits exist
                        continue
                    results.append(self._documents[idx])
                    scores.append(float(score))
                    indices.append(int(idx))

            return {"status": "OK", "results": results, "scores": scores,
                    "indices": indices, "reason": ""}
        except Exception as e:
            return {"status": "Failure", "results": [], "scores": [], "indices": [],
                    "reason": str(e)}

class ReRanker:
    # Larger MPNet-based model trained specifically for semantic search,
    # and designed to be scored with dot product.
    DEFAULT_MODEL = "multi-qa-mpnet-base-dot-v1"

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._load_error = ""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        except Exception as e:
            self._load_error = str(e)

    def rerank(self, query: str, documents: List[str], k: int) -> dict:
        """Rank documents against the query using dot-product of embeddings."""
        try:
            if self._model is None:
                return {"status": "Failure", "indices": [], "scores": [],
                        "reason": f"Model failed to load: {self._load_error}"}
            if not isinstance(query, str) or not query.strip():
                return {"status": "Failure", "indices": [], "scores": [],
                        "reason": "Query must be a non-empty string."}

            docs = [d for d in documents if isinstance(d, str)] if documents is not None else []
            if len(docs) != len(list(documents or [])):
                return {"status": "Failure", "indices": [], "scores": [],
                        "reason": "All documents must be strings."}
            if not docs:
                return {"status": "Failure", "indices": [], "scores": [],
                        "reason": "No documents provided."}
            if not isinstance(k, int) or k <= 0:
                return {"status": "Failure", "indices": [], "scores": [],
                        "reason": "k must be a positive integer."}

            k = min(k, len(docs))

            # Query embedding: shape (1, d); document embeddings: shape (n, d)
            q_emb = self._model.encode(
                [query], convert_to_numpy=True, show_progress_bar=False
            )
            d_emb = self._model.encode(
                docs, convert_to_numpy=True, show_progress_bar=False
            )

            # (1, d) @ (d, n) -> (1, n) -> score vector of length n
            score_vec = (q_emb @ d_emb.T)[0]

            # Rank descending and keep the top k
            order = np.argsort(-score_vec)[:k]

            return {
                "status": "OK",
                "indices": [int(i) for i in order],
                "scores": [float(score_vec[i]) for i in order],
                "reason": "",
            }
        except Exception as e:
            return {"status": "Failure", "indices": [], "scores": [], "reason": str(e)}
if __name__ == "__main__":
    logger.info("Running ETL")
    etl = ETL("scifact")
    docs = etl.load("scifact")  # assumed: list of strings
    logger.debug("Sample docs = {}".format(docs[:10]))

    q = "What are the side effects of B12 deficiency ?"

    # --- retriever ---
    retriever = Retriever(backend="faiss")

    status = retriever.initialize_backend()
    if status["status"] != "OK":
        raise RuntimeError(f"Retriever init failed: {status['Reason']}")

    status = retriever.build_index(docs)
    if status["status"] != "OK":
        raise RuntimeError(f"Index build failed: {status['Reason']}")

    r = retriever.search(q, k=10)
    if r["status"] != "OK":
        raise RuntimeError(f"Search failed: {r['reason']}")
    logger.debug("Retrieved indices = {}".format(r["indices"]))

    # --- reranker ---
    reranker = ReRanker()
    rr = reranker.rerank(q, r["results"], k=3)
    if rr["status"] != "OK":
        raise RuntimeError(f"Rerank failed: {rr['reason']}")

    for rank, (i, score) in enumerate(zip(rr["indices"], rr["scores"]), start=1):
        logger.info(
            f"#{rank} score={score:.4f} corpus_idx={r['indices'][i]} | {r['results'][i][:200]}"
        )
