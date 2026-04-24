"""
MODULE 6: Re-ranker
Re-ranks initial retrieval results
"""

from typing import List
from collections import defaultdict
from modules.models import RetrievedDocument


class ReRanker:
    """
    Re-ranks initial retrieval results
    
    Strategies:
    1. Diversity: Ensure varied sources
    2. Hybrid fusion: Combine multiple signals
    """
    
    def __init__(self, diversity_weight: float = 0.3):
        self.diversity_weight = diversity_weight
    
    def rerank(self, docs: List[RetrievedDocument], k: int = 5) -> List[RetrievedDocument]:
        """
        Re-rank by:
        1. Combined score (70%)
        2. Source diversity bonus (30%)
        """
        source_counts = defaultdict(int)
        
        scored_docs = []
        for doc in docs:
            diversity_bonus = 1.0 / (1 + source_counts[doc.source])
            final_score = (0.7 * doc.combined_score) + (0.3 * diversity_bonus * doc.combined_score)
            
            scored_docs.append((final_score, doc))
            source_counts[doc.source] += 1
        
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        
        reranked = []
        for i, (score, doc) in enumerate(scored_docs[:k]):
            doc.rank = i + 1
            reranked.append(doc)
        
        return reranked
