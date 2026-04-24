"""
PART B: CUSTOM RETRIEVAL SYSTEM - MAIN SCRIPT
Runs the modular RAG retrieval pipeline
"""

import json
from datetime import datetime
from modules.retrieval_engine import RetrievalEngine
from modules.failure_analyzer import FailureAnalyzer


def main():
    print("="*70)
    print("PART B: CUSTOM RETRIEVAL SYSTEM (MODULAR)")
    print("="*70)
    
    # Load chunks from Part A
    print("\n[SETUP] Loading chunks...")
    with open('processed_chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    print(f"[SETUP] Loaded {len(chunks)} chunks")
    
    # Initialize and index
    engine = RetrievalEngine()
    engine.index(chunks)
    
    # Test queries
    test_queries = [
        "What were the election results in Greater Accra region?",
        "How much was allocated to education in the 2025 budget?",
        "What is the GDP growth rate mentioned in the budget?",
        "Which party won the most seats in the election?",
        "Who won?",  # Adversarial
        "Money for schools",  # Informal
    ]
    
    print("\n" + "="*70)
    print("RETRIEVAL TESTING")
    print("="*70)
    
    all_results = {}
    for query in test_queries:
        print("\n" + "-"*60)
        
        # Hybrid-only retrieval (no expansion)
        print(f"\n[HYBRID-ONLY] {query}")
        results = engine.retrieve(query, k=3)
        
        all_results[query] = {
            'results': [{'text': d.text[:150], 'score': d.combined_score, 'source': d.source} for d in results]
        }
        
        if results:
            print(f"\n→ Top: {results[0].text[:100]}... (score: {results[0].combined_score:.3f})")
    
    # Failure analysis
    analyzer = FailureAnalyzer(engine)
    failures = analyzer.run_tests()
    
    # Demonstrate fixes
    fix_results = analyzer.demonstrate_fixes()
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'embedding_model': 'all-MiniLM-L6-v2',
        'vector_store': 'ChromaDB',
        'total_chunks': len(chunks),
        'features': ['hybrid_search', 'reranking', 'failure_analysis', 'fix_implementation'],  # query_expansion disabled
        'retrieval_results': all_results,
        'failure_analysis': failures,
        'fix_demonstration': fix_results
    }
    
    with open('part_b_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"""
✅ Modules: 8 separate files in /modules/
✅ Embedding: Sentence Transformers (384-dim)
✅ Vector Store: ChromaDB (persistent)
✅ Top-k Retrieval: Configurable with similarity scores
✅ Hybrid Search: Vector (α=0.7) + Keyword (0.3)  ← PRIMARY EXTENSION
⚠️  Query Expansion: Disabled (using hybrid-only)
✅ Re-ranking: Diversity-aware fusion
✅ Failure Analysis: {failures['failure_rate']:.0%} failure rate

Output: part_b_results.json
    """)
    print("="*70)
    print("PART B COMPLETE - Ready for Part C")
    print("="*70)
    
    return engine


if __name__ == "__main__":
    engine = main()
