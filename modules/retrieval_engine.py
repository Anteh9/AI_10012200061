"""
MODULE 7: Retrieval Engine
Orchestrates all modules in perfect synchrony
"""

import numpy as np
from typing import List, Dict
from modules.models import RetrievedDocument
from modules.embedding import EmbeddingPipeline
from modules.vector_store import VectorStore
from modules.query_expansion import QueryExpander
from modules.keyword_search import KeywordSearcher
from modules.reranker import ReRanker


class RetrievalEngine:
    """
    Orchestrates all modules:
    Query → Expand → Vector Search → Keyword Score → Hybrid Combine → Re-rank
    """
    
    def __init__(self):
        self.embedder = EmbeddingPipeline()
        self.vector_store = VectorStore()
        self.expander = QueryExpander()
        self.keyword_searcher = KeywordSearcher()
        self.reranker = ReRanker()
        self.corpus_chunks = []
    
    def index(self, chunks: List[Dict]):
        """Index all chunks for retrieval"""
        print("\n" + "="*60)
        print("INDEXING DOCUMENTS")
        print("="*60)
        
        self.corpus_chunks = chunks
        
        # Build embeddings
        texts = [c['text'] for c in chunks]
        embeddings = self.embedder.embed(texts)
        
        # Store in ChromaDB
        self.vector_store.add_chunks(chunks, embeddings)
        
        # Build keyword index
        self.keyword_searcher.build_index(texts)
        
        print(f"[ENGINE] Indexed {len(chunks)} chunks")
    
    def retrieve(self, query: str, k: int = 10, use_expansion: bool = False, 
                 hybrid_alpha: float = 0.7, metadata_filter: str = None) -> List[RetrievedDocument]:
        """
        Full retrieval pipeline with metadata filtering
        
        IMPROVEMENTS:
        - Default k increased from 5 to 10 (more results = better recall)
        - Added metadata_filter parameter (e.g., "education" sector tag)
        
        Args:
            k: Final number of results (default: 10 for better coverage)
            use_expansion: Enable query expansion
            hybrid_alpha: Vector weight (1-alpha = keyword weight)
            metadata_filter: Filter by sector tag (e.g., "education", "health")
        """
        print(f"\n[QUERY] '{query}'")
        print(f"[CONFIG] k={k}, expansion={use_expansion}, alpha={hybrid_alpha}")
        
        # Step 1: Expand query (disabled by default - using hybrid-only)
        if use_expansion:
            queries = self.expander.expand(query)
            print(f"[EXPANSION] {len(queries)} variants")
        else:
            queries = [query]
            print("[MODE] Hybrid search (vector + keyword)")
        
        # Step 2: Retrieve candidates
        candidates = {}
        
        for q in queries:
            emb = self.embedder.embed_query(q)
            results = self.vector_store.search(emb, k=k*3)
            
            for r in results:
                cid = r['id']
                if cid not in candidates:
                    candidates[cid] = {'vector_scores': [], 'doc': r}
                candidates[cid]['vector_scores'].append(r['score'])
        
        # Aggregate scores
        candidate_ids = list(candidates.keys())
        vector_scores = {cid: np.mean(candidates[cid]['vector_scores']) 
                        for cid in candidate_ids}
        
        # Step 3: Keyword scoring
        id_to_idx = {f"c{i}": i for i in range(len(self.corpus_chunks))}
        candidate_indices = [id_to_idx[cid] for cid in candidate_ids if cid in id_to_idx]
        keyword_scores = self.keyword_searcher.search(query, candidate_indices)
        
        # Step 4: Hybrid combination
        docs = []
        for i, cid in enumerate(candidate_ids):
            if cid not in id_to_idx:
                continue
            
            idx = id_to_idx[cid]
            chunk = self.corpus_chunks[idx]
            v_score = vector_scores[cid]
            k_score = keyword_scores.get(idx, 0)
            
            # Normalize and combine
            v_norm = max(0, v_score)
            k_norm = min(1.0, k_score * 2)
            combined = (hybrid_alpha * v_norm) + ((1 - hybrid_alpha) * k_norm)
            
            docs.append(RetrievedDocument(
                text=chunk['text'],
                source=chunk['source'],
                chunk_id=chunk['chunk_id'],
                vector_score=float(v_norm),
                keyword_score=float(k_norm),
                combined_score=float(combined),
                metadata=chunk.get('metadata', {})
            ))
        
        # Step 4.5: Metadata filtering (if specified)
        if metadata_filter:
            docs = [d for d in docs if metadata_filter.lower() in 
                   [t.lower() for t in d.metadata.get('tags', [])]]
            print(f"[FILTER] Filtered by tag '{metadata_filter}': {len(docs)} matches")
        
        # Step 5: Re-rank
        reranked = self.reranker.rerank(docs, k=k)
        
        print(f"[RESULT] Top {len(reranked)} documents")
        return reranked
