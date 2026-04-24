"""
PART A: DATA ENGINEERING & PREPARATION
This script handles:
1. Data cleaning for CSV and PDF datasets
2. Chunking strategy design with justification
3. Comparative analysis of chunking impact on retrieval quality
"""

import pandas as pd
import re
import nltk
from typing import List, Dict, Tuple, Any
import PyPDF2
import io
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
from dataclasses import dataclass
from datetime import datetime

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


@dataclass
class ChunkConfig:
    """Configuration for chunking strategies"""
    chunk_size: int  # in characters
    overlap: int     # in characters
    strategy_name: str
    description: str


@dataclass
class TextChunk:
    """Represents a text chunk with metadata"""
    text: str
    source: str
    chunk_id: int
    chunk_config: ChunkConfig
    metadata: Dict[str, Any]


class DataCleaner:
    """Handles data cleaning for both CSV and PDF data"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing special characters, extra spaces, and normalizing"""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove special characters but keep alphanumeric and basic punctuation
        text = re.sub(r'[^\w\s.,;:!?()-]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Normalize case (keep original for context, but we could lowercase)
        return text.strip()
    
    @staticmethod
    def clean_election_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean Ghana Election Results CSV data"""
        print("\n[1] CLEANING ELECTION DATA")
        print("=" * 50)
        
        # Display original info
        print(f"Original shape: {df.shape}")
        print(f"Original columns: {df.columns.tolist()}")
        print(f"Missing values:\n{df.isnull().sum()}")
        
        # Create a copy to avoid modifying original
        df_clean = df.copy()
        
        # Remove rows where all values are missing
        df_clean = df_clean.dropna(how='all')
        
        # Convert column names to lowercase and strip whitespace
        df_clean.columns = df_clean.columns.str.lower().str.strip()
        
        # Clean string columns
        string_columns = df_clean.select_dtypes(include=['object']).columns
        for col in string_columns:
            df_clean[col] = df_clean[col].apply(DataCleaner.clean_text)
        
        # Remove duplicates
        before_dedup = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        after_dedup = len(df_clean)
        print(f"Removed {before_dedup - after_dedup} duplicate rows")
        
        # Handle missing values based on column type
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(0)
        
        # Fill remaining string NaN with empty string
        string_cols = df_clean.select_dtypes(include=['object']).columns
        for col in string_cols:
            df_clean[col] = df_clean[col].fillna('')
        
        print(f"\nCleaned shape: {df_clean.shape}")
        print(f"Missing values after cleaning:\n{df_clean.isnull().sum()}")
        
        return df_clean
    
    @staticmethod
    def extract_pdf_text(pdf_path: str) -> str:
        """Extract text from local PDF file"""
        print("\n[2] EXTRACTING PDF TEXT")
        print("=" * 50)
        print(f"Reading local PDF: {pdf_path}")
        
        try:
            # Read PDF file locally
            pdf_file = open(pdf_path, 'rb')
            
            # Read PDF
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            print(f"Total pages: {num_pages}")
            
            text_parts = []
            for i, page in enumerate(pdf_reader.pages):
                if i % 10 == 0:
                    print(f"Processing page {i+1}/{num_pages}...")
                
                page_text = page.extract_text()
                if page_text:
                    # Clean the extracted text
                    cleaned_text = DataCleaner.clean_text(page_text)
                    if cleaned_text:
                        text_parts.append(cleaned_text)
            
            pdf_file.close()
            
            full_text = ' '.join(text_parts)
            print(f"\nExtracted {len(full_text)} characters from PDF")
            print(f"Extracted {len(full_text.split())} words from PDF")
            
            return full_text
            
        except Exception as e:
            print(f"Error extracting PDF: {e}")
            return ""
    
    @staticmethod
    def clean_budget_text(text: str) -> str:
        """Additional cleaning specific to budget documents"""
        print("\n[3] CLEANING BUDGET TEXT")
        print("=" * 50)
        
        # Remove page numbers (common patterns)
        text = re.sub(r'\n\s*\d+\s*\n', '\n', text)  # Standalone page numbers
        text = re.sub(r'Page\s+\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)
        
        # Remove common headers/footers patterns
        text = re.sub(r'\d{4}\s+Budget\s+Statement.*?(?=\n)', '', text, flags=re.IGNORECASE)
        
        # Remove multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up section numbers and formatting
        text = re.sub(r'\n\s*\d+\.\d+\s+', '\n', text)
        
        print(f"Final cleaned text: {len(text)} characters")
        
        return text


class ChunkingEngine:
    """Implements and compares different chunking strategies"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define chunking configurations for comparison
        self.configs = [
            ChunkConfig(
                chunk_size=500,
                overlap=50,
                strategy_name="small_chunks",
                description="Small chunks (500 chars) with minimal overlap - good for precise retrieval"
            ),
            ChunkConfig(
                chunk_size=1000,
                overlap=100,
                strategy_name="medium_chunks",
                description="Medium chunks (1000 chars) with standard overlap - balanced approach"
            ),
            ChunkConfig(
                chunk_size=2000,
                overlap=200,
                strategy_name="large_chunks",
                description="Large chunks (2000 chars) with more overlap - better context preservation"
            ),
            ChunkConfig(
                chunk_size=1000,
                overlap=300,
                strategy_name="high_overlap",
                description="High overlap strategy (1000 chars, 300 overlap) - smoother transitions"
            )
        ]
    
    def chunk_text(self, text: str, config: ChunkConfig, source: str, 
                   base_metadata: Dict = None) -> List[TextChunk]:
        """
        IMPROVED: Sentence-aware chunking that never cuts sentences mid-way.
        Also extracts metadata tags (sector, topic) for filtering.
        
        Justification:
        1. Sentence boundaries = natural semantic units
        2. Cutting mid-sentence hurts retrieval quality (incomplete context)
        3. Metadata tags enable sector filtering (e.g., "education")
        """
        if base_metadata is None:
            base_metadata = {}
        
        # IMPROVEMENT 1: Split into sentences first (never cut mid-sentence)
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        chunk_id = 0
        current_chunk_sentences = []
        current_chunk_length = 0
        char_position = 0
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_chunk_length + sentence_len > config.chunk_size and current_chunk_sentences:
                chunk_text = ' '.join(current_chunk_sentences)
                
                # IMPROVEMENT 2: Extract metadata tags from content
                metadata_tags = self._extract_metadata_tags(chunk_text)
                
                metadata = {
                    **base_metadata,
                    'char_start': char_position - current_chunk_length,
                    'char_end': char_position,
                    'chunk_length': len(chunk_text),
                    'sentence_count': len(current_chunk_sentences),
                    'tags': metadata_tags,  # NEW: sector/topic tags
                    'created_at': datetime.now().isoformat()
                }
                
                chunks.append(TextChunk(
                    text=chunk_text,
                    source=source,
                    chunk_id=chunk_id,
                    chunk_config=config,
                    metadata=metadata
                ))
                chunk_id += 1
                
                # IMPROVEMENT 3: Handle overlap by keeping some sentences
                overlap_sentences = []
                overlap_length = 0
                # Keep sentences from the end for overlap
                for s in reversed(current_chunk_sentences):
                    if overlap_length + len(s) <= config.overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s) + 1  # +1 for space
                    else:
                        break
                
                current_chunk_sentences = overlap_sentences
                current_chunk_length = overlap_length
            
            # Add current sentence to chunk
            current_chunk_sentences.append(sentence)
            current_chunk_length += sentence_len + 1  # +1 for space
            char_position += sentence_len + 1
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            metadata_tags = self._extract_metadata_tags(chunk_text)
            
            metadata = {
                **base_metadata,
                'char_start': char_position - current_chunk_length,
                'char_end': char_position,
                'chunk_length': len(chunk_text),
                'sentence_count': len(current_chunk_sentences),
                'tags': metadata_tags,
                'created_at': datetime.now().isoformat()
            }
            
            chunks.append(TextChunk(
                text=chunk_text,
                source=source,
                chunk_id=chunk_id,
                chunk_config=config,
                metadata=metadata
            ))
        
        return chunks
    
    def _extract_metadata_tags(self, text: str) -> List[str]:
        """
        IMPROVEMENT: Extract sector/topic tags from text for metadata filtering.
        E.g., "education", "health", "agriculture", "infrastructure"
        """
        text_lower = text.lower()
        tags = []
        
        # Sector keywords for Ghana budget
        sector_keywords = {
            'education': ['education', 'school', 'university', 'student', 'teacher', 'textbook', 'curriculum', 'free shs', 'scholarship'],
            'health': ['health', 'hospital', 'medical', 'doctor', 'nurse', 'pharmaceutical', 'malaria', 'nhis'],
            'agriculture': ['agriculture', 'farmer', 'crop', 'fishing', 'livestock', 'planting for food'],
            'infrastructure': ['road', 'bridge', 'transport', 'railway', 'port', 'airport', 'infrastructure'],
            'energy': ['energy', 'electricity', 'power', 'solar', 'grid', 'eca'],
            'finance': ['budget', 'revenue', 'tax', 'imf', 'debt', 'gdp', 'fiscal', 'monetary'],
            'elections': ['election', 'vote', 'candidate', 'party', 'npp', 'ndc', 'parliament']
        }
        
        for sector, keywords in sector_keywords.items():
            if any(kw in text_lower for kw in keywords):
                tags.append(sector)
        
        return tags
    
    def chunk_election_data(self, df: pd.DataFrame, config: ChunkConfig) -> List[TextChunk]:
        """Convert election DataFrame into meaningful text chunks"""
        print(f"\n[4] CHUNKING ELECTION DATA with {config.strategy_name}")
        print("=" * 50)
        
        chunks = []
        
        # Create structured text from each row
        for idx, row in df.iterrows():
            # Convert row to structured text
            row_text = f"Election Result Row {idx + 1}: "
            row_parts = []
            
            for col in df.columns:
                if row[col] and str(row[col]).strip():
                    row_parts.append(f"{col}: {row[col]}")
            
            if row_parts:
                row_text += ". ".join(row_parts)
                
                # Create chunk from this row
                metadata = {
                    'row_index': idx,
                    'data_type': 'election_result',
                    'columns_present': [col for col in df.columns if row[col]]
                }
                
                # If row text is too long, split it
                if len(row_text) > config.chunk_size:
                    sub_chunks = self.chunk_text(
                        row_text, config, "election_csv", metadata
                    )
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(TextChunk(
                        text=row_text,
                        source="election_csv",
                        chunk_id=len(chunks),
                        chunk_config=config,
                        metadata=metadata
                    ))
        
        print(f"Created {len(chunks)} chunks from election data")
        return chunks
    
    def chunk_budget_text(self, text: str, config: ChunkConfig) -> List[TextChunk]:
        """Chunk budget PDF text"""
        print(f"\n[5] CHUNKING BUDGET TEXT with {config.strategy_name}")
        print("=" * 50)
        
        metadata = {
            'data_type': 'budget_document',
            'total_length': len(text),
            'estimated_pages': len(text) // 3000  # Rough estimate
        }
        
        chunks = self.chunk_text(text, config, "budget_pdf", metadata)
        print(f"Created {len(chunks)} chunks from budget text")
        
        return chunks
    
    def evaluate_chunking_quality(self, chunks: List[TextChunk], 
                                   sample_queries: List[str]) -> Dict:
        """
        Evaluate chunking quality by measuring:
        1. Average chunk size
        2. Coverage of queries
        3. Embedding similarity variance
        
        This helps justify chunking strategy selection.
        """
        print(f"\n[6] EVALUATING CHUNKING QUALITY")
        print("=" * 50)
        
        # Calculate basic metrics
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        
        metrics = {
            'num_chunks': len(chunks),
            'avg_chunk_size': np.mean(chunk_lengths),
            'std_chunk_size': np.std(chunk_lengths),
            'min_chunk_size': np.min(chunk_lengths),
            'max_chunk_size': np.max(chunk_lengths),
            'total_chars': sum(chunk_lengths)
        }
        
        print(f"Number of chunks: {metrics['num_chunks']}")
        print(f"Average chunk size: {metrics['avg_chunk_size']:.1f} chars")
        print(f"Chunk size std dev: {metrics['std_chunk_size']:.1f}")
        
        # Test retrieval quality with sample queries
        if sample_queries:
            print(f"\nTesting with {len(sample_queries)} sample queries...")
            
            # Embed chunks
            chunk_texts = [chunk.text for chunk in chunks]
            chunk_embeddings = self.embedding_model.encode(chunk_texts)
            
            query_scores = []
            for query in sample_queries:
                query_embedding = self.embedding_model.encode([query])
                similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
                top_3_scores = sorted(similarities, reverse=True)[:3]
                query_scores.append({
                    'query': query,
                    'top_3_avg': np.mean(top_3_scores),
                    'score_variance': np.var(similarities)
                })
            
            metrics['sample_query_performance'] = query_scores
            
            # Calculate average similarity score
            avg_top_3 = np.mean([q['top_3_avg'] for q in query_scores])
            print(f"Average top-3 similarity score: {avg_top_3:.3f}")
        
        return metrics


class ChunkingAnalyzer:
    """Compares different chunking strategies and provides analysis"""
    
    def __init__(self):
        self.engine = ChunkingEngine()
        self.results = {}
    
    def compare_strategies(self, election_df: pd.DataFrame, budget_text: str,
                          sample_queries: List[str]) -> Dict:
        """Run comparative analysis of all chunking strategies"""
        print("\n" + "=" * 70)
        print("COMPARATIVE CHUNKING ANALYSIS")
        print("=" * 70)
        
        comparison_results = {}
        
        for config in self.engine.configs:
            print(f"\n\n--- Testing {config.strategy_name} ---")
            print(config.description)
            
            # Chunk election data
            election_chunks = self.engine.chunk_election_data(election_df, config)
            
            # Chunk budget data
            budget_chunks = self.engine.chunk_budget_text(budget_text, config)
            
            # Combine all chunks
            all_chunks = election_chunks + budget_chunks
            
            # Evaluate quality
            metrics = self.engine.evaluate_chunking_quality(all_chunks, sample_queries)
            
            # Store results
            comparison_results[config.strategy_name] = {
                'config': config,
                'metrics': metrics,
                'chunks': all_chunks
            }
        
        return comparison_results
    
    def recommend_strategy(self, comparison_results: Dict) -> Tuple[str, ChunkConfig]:
        """Recommend best chunking strategy based on analysis"""
        print("\n" + "=" * 70)
        print("CHUNKING STRATEGY RECOMMENDATION")
        print("=" * 70)
        
        # Score each strategy (lower is better)
        strategy_scores = {}
        
        for name, result in comparison_results.items():
            metrics = result['metrics']
            
            # Factors to consider:
            # 1. Not too many chunks (efficiency)
            chunk_count_score = metrics['num_chunks'] / 100
            
            # 2. Reasonable chunk size (not too small for context, not too large for precision)
            # Optimal range: 800-1200 characters
            size_deviation = abs(metrics['avg_chunk_size'] - 1000) / 1000
            
            # 3. Low variance (consistent chunks)
            variance_score = metrics['std_chunk_size'] / 500
            
            # 4. Good retrieval performance (if tested)
            if 'sample_query_performance' in metrics:
                retrieval_score = 1 - np.mean([q['top_3_avg'] for q in metrics['sample_query_performance']])
            else:
                retrieval_score = 0.5
            
            # Combined score
            total_score = chunk_count_score + size_deviation + variance_score + retrieval_score
            strategy_scores[name] = total_score
            
            print(f"\n{name}:")
            print(f"  Total score: {total_score:.3f}")
            print(f"  Chunks: {metrics['num_chunks']}")
            print(f"  Avg size: {metrics['avg_chunk_size']:.1f}")
        
        # Find best strategy
        best_strategy = min(strategy_scores, key=strategy_scores.get)
        best_config = comparison_results[best_strategy]['config']
        best_chunks = comparison_results[best_strategy]['chunks']
        
        print(f"\n{'='*70}")
        print(f"RECOMMENDED STRATEGY: {best_strategy}")
        print(f"{'='*70}")
        print(f"Configuration: {best_config.chunk_size} chars, {best_config.overlap} overlap")
        print(f"Description: {best_config.description}")
        print(f"Total chunks created: {len(best_chunks)}")
        
        return best_strategy, best_config, best_chunks
    
    def save_chunks(self, chunks: List[TextChunk], output_file: str):
        """Save chunks to JSON for next stages"""
        print(f"\nSaving chunks to {output_file}...")
        
        chunk_data = []
        for chunk in chunks:
            chunk_data.append({
                'text': chunk.text,
                'source': chunk.source,
                'chunk_id': chunk.chunk_id,
                'chunk_config': {
                    'chunk_size': chunk.chunk_config.chunk_size,
                    'overlap': chunk.chunk_config.overlap,
                    'strategy_name': chunk.chunk_config.strategy_name
                },
                'metadata': chunk.metadata
            })
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(chunk_data)} chunks to {output_file}")


def main():
    """Main execution for Part A"""
    print("=" * 70)
    print("PART A: DATA ENGINEERING & PREPARATION")
    print("=" * 70)
    
    # Initialize components
    cleaner = DataCleaner()
    analyzer = ChunkingAnalyzer()
    
    # Local file paths for datasets
    ELECTION_CSV_PATH = "Ghana_Election_Result.csv"
    BUDGET_PDF_PATH = "2025-Budget-Statement-and-Economic-Policy_v4.pdf"
    
    # Sample queries for evaluation
    sample_queries = [
        "What were the election results in Greater Accra region?",
        "How much was allocated to education in the 2025 budget?",
        "What is the GDP growth rate mentioned in the budget?",
        "Which party won the most seats in the election?",
        "What are the main economic policies for 2025?"
    ]
    
    try:
        # [STEP 1] Load and clean election data
        print("\n\n[STEP 1] LOADING ELECTION DATA")
        print("=" * 50)
        print(f"Loading local file: {ELECTION_CSV_PATH}")
        
        election_df = pd.read_csv(ELECTION_CSV_PATH)
        election_df_clean = cleaner.clean_election_data(election_df)
        
        # Save cleaned data
        election_df_clean.to_csv('cleaned_election_data.csv', index=False)
        print("\nCleaned election data saved to: cleaned_election_data.csv")
        
    except Exception as e:
        print(f"Error loading election data: {e}")
        print("Creating sample election data for testing...")
        
        # Create sample data for testing
        sample_data = {
            'region': ['Greater Accra', 'Ashanti', 'Western', 'Eastern', 'Central'],
            'constituency': ['Ablekuma North', 'Kumasi Central', 'Takoradi', 'Koforidua', 'Cape Coast'],
            'party': ['NDC', 'NPP', 'NPP', 'NDC', 'NDC'],
            'candidate': ['John Doe', 'Jane Smith', 'Kwame Mensah', 'Ama Darko', 'Kofi Addo'],
            'votes': [25000, 32000, 18000, 22000, 21000],
            'percentage': [45.2, 52.1, 38.5, 48.3, 46.7]
        }
        election_df_clean = pd.DataFrame(sample_data)
        election_df_clean.to_csv('cleaned_election_data.csv', index=False)
    
    try:
        # [STEP 2] Load and clean budget PDF
        print("\n\n[STEP 2] LOADING BUDGET PDF")
        print("=" * 50)
        print(f"Loading local file: {BUDGET_PDF_PATH}")
        
        budget_text = cleaner.extract_pdf_text(BUDGET_PDF_PATH)
        budget_text = cleaner.clean_budget_text(budget_text)
        
        # Save cleaned text
        with open('cleaned_budget_text.txt', 'w', encoding='utf-8') as f:
            f.write(budget_text)
        print("\nCleaned budget text saved to: cleaned_budget_text.txt")
        
    except Exception as e:
        print(f"Error loading budget PDF: {e}")
        print("Creating sample budget text for testing...")
        
        # Create sample budget text
        budget_text = """
        2025 BUDGET STATEMENT AND ECONOMIC POLICY OF GHANA
        
        1.0 INTRODUCTION
        The 2025 budget focuses on economic recovery and sustainable growth. The government 
        has allocated significant resources to key sectors including education, health, and infrastructure.
        
        2.0 ECONOMIC OVERVIEW
        Ghana's GDP growth is projected at 3.8% for 2025. Inflation has stabilized at 15.2% and 
        is expected to decline further. The government remains committed to fiscal discipline.
        
        3.0 EDUCATION SECTOR
        An amount of GH₵ 12.5 billion has been allocated to education. This includes:
        - Free Senior High School continuation: GH₵ 5.2 billion
        - Teacher training and development: GH₵ 2.1 billion
        - Infrastructure development: GH₵ 3.8 billion
        - Capitation grants: GH₵ 1.4 billion
        
        4.0 HEALTH SECTOR
        The health sector receives GH₵ 8.3 billion in allocation. Key initiatives include:
        - National Health Insurance Scheme expansion
        - Construction of new hospitals
        - Medical equipment procurement
        
        5.0 AGRICULTURE
        Agriculture gets GH₵ 4.2 billion to support:
        - Planting for Food and Jobs program
        - Fertilizer subsidies
        - Irrigation development
        """
        
        with open('cleaned_budget_text.txt', 'w', encoding='utf-8') as f:
            f.write(budget_text)
    
    # [STEP 3] Compare chunking strategies
    print("\n\n[STEP 3] COMPARING CHUNKING STRATEGIES")
    print("=" * 50)
    
    comparison_results = analyzer.compare_strategies(
        election_df_clean, 
        budget_text, 
        sample_queries
    )
    
    # [STEP 4] Get recommendation
    best_strategy, best_config, best_chunks = analyzer.recommend_strategy(comparison_results)
    
    # [STEP 5] Save chunks for Part B
    analyzer.save_chunks(best_chunks, 'processed_chunks.json')
    
    # [STEP 6] Generate analysis report
    report = {
        'timestamp': datetime.now().isoformat(),
        'election_data': {
            'source': ELECTION_CSV_PATH,
            'rows_cleaned': len(election_df_clean),
            'columns': election_df_clean.columns.tolist()
        },
        'budget_data': {
            'source': BUDGET_PDF_PATH,
            'characters_extracted': len(budget_text),
            'words_extracted': len(budget_text.split())
        },
        'chunking_comparison': {
            name: {
                'description': result['config'].description,
                'num_chunks': result['metrics']['num_chunks'],
                'avg_chunk_size': result['metrics']['avg_chunk_size'],
                'std_chunk_size': result['metrics']['std_chunk_size']
            }
            for name, result in comparison_results.items()
        },
        'recommended_strategy': {
            'name': best_strategy,
            'chunk_size': best_config.chunk_size,
            'overlap': best_config.overlap,
            'total_chunks': len(best_chunks)
        }
    }
    
    with open('part_a_analysis_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n\nAnalysis report saved to: part_a_analysis_report.json")
    
    # [STEP 7] Justification summary
    print("\n" + "=" * 70)
    print("DESIGN JUSTIFICATION SUMMARY")
    print("=" * 70)
    print("""
    CHUNKING STRATEGY JUSTIFICATION:
    
    1. Why Character-Based Chunking?
       - More predictable than token-based (works across different LLMs)
       - Easier to implement and debug
       - Consistent with embedding model expectations
    
    2. Why Overlap?
       - Prevents context loss at chunk boundaries
       - Ensures continuity of information
       - Critical for questions spanning multiple sentences
    
    3. Chunk Size Selection:
       - Too small (<500): Loses context, poor semantic understanding
       - Too large (>2000): Dilutes relevance, harder to retrieve precise info
       - Sweet spot (~1000): Balances context and precision
    
    4. Domain Suitability:
       - Election data: Row-based with structured text preserves relationships
       - Budget text: Sliding window captures continuous narrative
    
    5. Evaluation Metrics:
       - Average similarity scores for sample queries
       - Consistency in chunk sizes (low variance)
       - Total chunk count (efficiency)
    """)
    
    print("\n" + "=" * 70)
    print("PART A COMPLETE - Proceed to Part B")
    print("=" * 70)
    
    return best_chunks, comparison_results


if __name__ == "__main__":
    chunks, comparison = main()
