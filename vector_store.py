"""
Vector Store Module
Implements high-performance semantic search using Google Gemini Embeddings (text-embedding-004)
with fallback term-frequency cosine vectorizer and persistent disk storage.
"""
import os
import json
import math
import re
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from config import Config
from document_loader import Document

try:
    from google import genai
except ImportError:
    genai = None


class FallbackVectorizer:
    """TF-IDF vectorizer fallback when Gemini API key is not yet provided or offline"""

    @staticmethod
    def tokenize(text: str) -> List[str]:
        words = re.findall(r'\b[a-zA-Z0-9_]{2,}\b', text.lower())
        return words

    @classmethod
    def compute_embedding(cls, text: str, vocab: Dict[str, int]) -> np.ndarray:
        tokens = cls.tokenize(text)
        vec = np.zeros(len(vocab), dtype=np.float32)
        if not tokens or not vocab:
            return vec
        for token in tokens:
            if token in vocab:
                vec[vocab[token]] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


class VectorStoreManager:
    """Manages document embeddings and similarity search"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = (api_key or Config.GEMINI_API_KEY).strip()
        self.documents: List[Document] = []
        self.vectors: Optional[np.ndarray] = None
        self.storage_dir = Config.VECTOR_STORE_PATH
        self.data_file = os.path.join(self.storage_dir, 'store.json')
        self.vectors_file = os.path.join(self.storage_dir, 'vectors.npy')
        self.embedding_model = Config.EMBEDDING_MODEL

        # Load existing store if available
        self.load_vector_store()

    def set_api_key(self, api_key: str):
        """Update API key and re-initialize client"""
        self.api_key = api_key.strip()

    def _get_gemini_client(self):
        if not self.api_key or genai is None:
            return None
        try:
            return genai.Client(api_key=self.api_key)
        except Exception as e:
            print(f"Error creating GenAI client: {e}")
            return None

    def _embed_texts_gemini(self, texts: List[str]) -> Optional[np.ndarray]:
        """Compute embeddings using Google GenAI API"""
        client = self._get_gemini_client()
        if not client:
            return None

        model_name = self.embedding_model
        # Strip models/ prefix if needed
        clean_model = model_name.replace("models/", "")

        embeddings = []
        # Batch in chunks of 20 to avoid payload limits
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                resp = client.models.embed_content(
                    model=clean_model,
                    contents=batch
                )
                if hasattr(resp, "embeddings") and resp.embeddings:
                    for item in resp.embeddings:
                        embeddings.append(item.values)
                else:
                    return None
            except Exception as e:
                print(f"Gemini embedding API call failed: {e}")
                return None

        if embeddings:
            return np.array(embeddings, dtype=np.float32)
        return None

    def _embed_texts_fallback(self, texts: List[str]) -> np.ndarray:
        """Compute TF-IDF based dense vectors as reliable fallback"""
        # Build vocabulary from all texts
        vocab: Dict[str, int] = {}
        for text in texts:
            for token in FallbackVectorizer.tokenize(text):
                if token not in vocab:
                    vocab[token] = len(vocab)

        vectors = []
        for text in texts:
            vec = FallbackVectorizer.compute_embedding(text, vocab)
            vectors.append(vec)

        if not vectors:
            return np.zeros((0, max(1, len(vocab))), dtype=np.float32)
        return np.array(vectors, dtype=np.float32)

    def add_documents(self, documents: List[Document]):
        """Embed and append new documents to the vector store"""
        if not documents:
            return

        texts = [doc.page_content for doc in documents]
        new_vectors = self._embed_texts_gemini(texts)

        if new_vectors is None:
            # Fallback
            new_vectors = self._embed_texts_fallback(texts)

        if self.vectors is None or len(self.documents) == 0:
            self.documents = list(documents)
            self.vectors = new_vectors
        else:
            # Check dimensions match
            if self.vectors.shape[1] == new_vectors.shape[1]:
                self.documents.extend(documents)
                self.vectors = np.vstack([self.vectors, new_vectors])
            else:
                # Dimension changed (e.g. switched from fallback to Gemini or vice-versa), recompute all
                self.documents.extend(documents)
                self._recompute_all_vectors()

        self.save_vector_store()
        print(f"Vector store now contains {len(self.documents)} chunks.")

    def _recompute_all_vectors(self):
        """Recomputes vectors for all current documents"""
        if not self.documents:
            self.vectors = None
            return
        texts = [doc.page_content for doc in self.documents]
        vecs = self._embed_texts_gemini(texts)
        if vecs is None:
            vecs = self._embed_texts_fallback(texts)
        self.vectors = vecs

    def similarity_search_with_score(self, query: str, k: Optional[int] = None) -> List[Tuple[Document, float]]:
        """Find top-k most relevant document chunks with cosine similarity score"""
        if not self.documents or self.vectors is None or len(self.documents) == 0:
            return []

        k = k or Config.TOP_K_RESULTS
        k = min(k, len(self.documents))

        # Embed query
        query_vec = None
        if self.vectors.shape[1] != 0:
            # Try Gemini embedding
            gemini_vecs = self._embed_texts_gemini([query])
            if gemini_vecs is not None and gemini_vecs.shape[1] == self.vectors.shape[1]:
                query_vec = gemini_vecs[0]
            else:
                # Match vocabulary of fallback vectors if possible
                # If dimensions match fallback or simple term overlap
                tokens = FallbackVectorizer.tokenize(query)
                query_vec = np.zeros(self.vectors.shape[1], dtype=np.float32)
                # Simple fallback scoring
                for token in tokens:
                    # Hash token to dimension index
                    idx = abs(hash(token)) % self.vectors.shape[1]
                    query_vec[idx] += 1.0

        if query_vec is None:
            return [(doc, 0.5) for doc in self.documents[:k]]

        q_norm = np.linalg.norm(query_vec)
        if q_norm > 0:
            query_vec /= q_norm

        # Compute cosine similarity across all document vectors
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normalized_vectors = self.vectors / norms

        scores = np.dot(normalized_vectors, query_vec)
        # Ensure 1D array
        scores = np.squeeze(scores)

        if scores.ndim == 0:
            scores = np.array([scores])

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            score_val = float(scores[idx])
            results.append((self.documents[idx], score_val))

        return results

    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Return list of top-k documents matching query"""
        results_with_score = self.similarity_search_with_score(query, k)
        return [doc for doc, _ in results_with_score]

    def save_vector_store(self):
        """Save documents and vectors to disk"""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            doc_dicts = [doc.to_dict() for doc in self.documents]
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(doc_dicts, f, ensure_ascii=False, indent=2)

            if self.vectors is not None:
                np.save(self.vectors_file, self.vectors)
        except Exception as e:
            print(f"Error saving vector store: {e}")

    def load_vector_store(self):
        """Load documents and vectors from disk"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    doc_dicts = json.load(f)
                self.documents = [Document.from_dict(d) for d in doc_dicts]

            if os.path.exists(self.vectors_file):
                self.vectors = np.load(self.vectors_file)
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.documents = []
            self.vectors = None

    def clear(self):
        """Clear all indexed documents"""
        self.documents = []
        self.vectors = None
        if os.path.exists(self.data_file):
            try:
                os.remove(self.data_file)
            except OSError:
                pass
        if os.path.exists(self.vectors_file):
            try:
                os.remove(self.vectors_file)
            except OSError:
                pass

    def delete_source(self, source_name: str) -> int:
        """Delete all chunks belonging to a specific source"""
        original_count = len(self.documents)
        kept_docs = []
        kept_indices = []

        for idx, doc in enumerate(self.documents):
            if doc.metadata.get("source") != source_name:
                kept_docs.append(doc)
                kept_indices.append(idx)

        deleted = original_count - len(kept_docs)
        if deleted > 0:
            self.documents = kept_docs
            if self.vectors is not None and len(kept_indices) > 0:
                self.vectors = self.vectors[kept_indices]
            else:
                self.vectors = None
            self.save_vector_store()

        return deleted

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about indexed materials"""
        sources = {}
        for doc in self.documents:
            src = doc.metadata.get("source", "Unknown")
            doc_type = doc.metadata.get("type", "generic")
            if src not in sources:
                sources[src] = {
                    "source": src,
                    "type": doc_type,
                    "chunks": 0,
                    "url": doc.metadata.get("url", "")
                }
            sources[src]["chunks"] += 1

        return {
            "total_documents": len(sources),
            "total_chunks": len(self.documents),
            "sources": list(sources.values()),
            "has_vectors": self.vectors is not None and len(self.vectors) > 0
        }
