"""
MODULE 3: Vector Store
ChromaDB wrapper for persistent vector storage
"""

import numpy as np
import chromadb
from typing import List, Dict


class VectorStore:
    """
    ChromaDB wrapper for persistent vector storage
    
    Features:
    - Persistent storage (data survives restart)
    - Metadata filtering
    - Fast approximate search
    """
    
    def __init__(self, collection_name: str = "acity_rag"):
        # Use in-memory client for Streamlit Cloud compatibility
        # (Data is re-indexed on each app start from processed_chunks.json)
        self.client = chromadb.Client()
        
        # Reset collection if exists
        try:
            self.client.delete_collection(collection_name)
        except:
            pass
        
        self.collection = self.client.create_collection(name=collection_name)
        self.chunks = {}  # Local cache
        print(f"[VECTOR STORE] Collection '{collection_name}' ready (in-memory)")
    
    def add_chunks(self, chunks: List[Dict], embeddings: np.ndarray):
        """Store chunks with embeddings"""
        print(f"[VECTOR STORE] Storing {len(chunks)} chunks...")
        
        # Batch add
        ids = [f"c{i}" for i in range(len(chunks))]
        texts = [c['text'] for c in chunks]
        
        # Convert metadata to primitive types only (ChromaDB requirement)
        metadatas = []
        for c in chunks:
            meta = {'source': c['source'], 'chunk_id': c['chunk_id']}
            # Convert any lists to strings
            for key, val in c.get('metadata', {}).items():
                if isinstance(val, list):
                    meta[key] = ','.join(str(v) for v in val)
                elif isinstance(val, (str, int, float, bool)) or val is None:
                    meta[key] = val
                else:
                    meta[key] = str(val)
            metadatas.append(meta)
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        
        # Cache locally
        for i, chunk in enumerate(chunks):
            self.chunks[ids[i]] = chunk
        
        print(f"[VECTOR STORE] Stored successfully")
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Dict]:
        """Top-k similarity search with scores"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        # Format results
        docs = []
        for i in range(len(results['ids'][0])):
            docs.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'score': results['distances'][0][i],
                'metadata': results['metadatas'][0][i]
            })
        
        return docs
