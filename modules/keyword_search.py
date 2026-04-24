"""
MODULE 5: Keyword Search
TF-IDF based keyword matching for hybrid search
"""

from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordSearcher:
    """
    TF-IDF based keyword matching
    
    Purpose: Catch exact term matches that semantic search might miss
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.tfidf_matrix = None
        self.corpus = []
    
    def build_index(self, texts: List[str]):
        """Build TF-IDF index"""
        self.corpus = texts
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"[KEYWORD] Index built: {self.tfidf_matrix.shape}")
    
    def search(self, query: str, doc_indices: List[int]) -> Dict[int, float]:
        """Score specific documents by keyword match"""
        query_vec = self.vectorizer.transform([query])
        scores = {}
        
        for idx in doc_indices:
            score = (query_vec @ self.tfidf_matrix[idx].T).toarray()[0][0]
            scores[idx] = score
        
        return scores
