"""
MODULE 1: Data Models
Core data structures used across all modules
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class RetrievedDocument:
    """Standard format for retrieved documents across all modules"""
    text: str
    source: str
    chunk_id: int
    vector_score: float = 0.0
    keyword_score: float = 0.0
    combined_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    rank: int = 0
