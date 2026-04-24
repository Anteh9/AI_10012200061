"""
MODULE 2: Embedding Pipeline
Generates embeddings using Sentence Transformers
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingPipeline:
    """
    Generates embeddings using Sentence Transformers
    
    Design Choice: all-MiniLM-L6-v2
    - 384 dimensions (efficient storage)
    - L2 normalized for cosine similarity
    - Fast inference on CPU
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        print(f"\n[EMBEDDING] Loading: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        print(f"[EMBEDDING] Ready: {self.dim} dimensions")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Generate normalized embeddings"""
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    
    def embed_query(self, query: str) -> np.ndarray:
        """Single query embedding"""
        return self.embed([query])[0]
