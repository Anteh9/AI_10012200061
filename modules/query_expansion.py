"""
MODULE 4: Query Expansion
Expands queries to improve recall
"""

from typing import List


class QueryExpander:
    """
    Expands queries to improve recall
    
    Strategies:
    1. Synonym expansion (e.g., money → budget/allocation)
    2. Domain expansion (election → vote/party/candidate)
    """
    
    def __init__(self):
        self.synonyms = {
            'money': ['budget', 'allocation', 'funds', 'expenditure', 'revenue'],
            'won': ['elected', 'victory', 'seats', 'majority'],
            'schools': ['education', 'academic', 'learning'],
            'economy': ['gdp', 'fiscal', 'economic', 'growth'],
        }
        
        self.domain_terms = {
            'election': ['vote', 'candidate', 'party', 'npp', 'ndc', 'parliament'],
            'budget': ['fiscal', 'allocation', 'revenue', 'expenditure', 'mofep'],
            'region': ['greater accra', 'ashanti', 'western', 'volta'],
        }
    
    def expand(self, query: str) -> List[str]:
        """Generate expanded queries"""
        query_lower = query.lower()
        expanded = [query]  # Always keep original
        
        # Synonym expansion
        for word, syns in self.synonyms.items():
            if word in query_lower:
                for syn in syns[:2]:
                    expanded.append(query.replace(word, syn))
        
        # Domain expansion
        for domain, terms in self.domain_terms.items():
            if any(t in query_lower for t in terms[:2]):
                expanded.append(f"{query} {' '.join(terms[:3])}")
        
        return list(set(expanded))[:5]
