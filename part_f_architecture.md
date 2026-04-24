# PART F: ARCHITECTURE & SYSTEM DESIGN

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           ACADEMIC CITY RAG SYSTEM                          │
│                    (Ghana Election & Budget Query System)                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                              LAYER 1: DATA INPUT                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│   ┌─────────────────┐      ┌─────────────────┐                               │
│   │ Ghana Election  │      │  2025 Budget    │                               │
│   │   CSV Data      │      │    PDF Data     │                               │
│   │                 │      │                 │                               │
│   │ • Party votes   │      │ • Allocations   │                               │
│   │ • Regional      │      │ • Initiatives   │                               │
│   │   results       │      │ • Sectoral      │                               │
│   │ • 2020 data     │      │   breakdowns    │                               │
│   └────────┬────────┘      └────────┬────────┘                               │
│            │                        │                                        │
│            └────────────┬───────────┘                                        │
│                         │                                                    │
│                         ▼                                                    │
│            ┌─────────────────────┐                                           │
│            │   PART A: DATA      │                                           │
│            │   PREPARATION       │                                           │
│            │                     │                                           │
│            │ • Data cleaning     │                                           │
│            │ • Sentence-aware    │◄─── IMPROVEMENT: No mid-sentence cuts   │
│            │   chunking          │                                           │
│            │ • Metadata          │◄─── IMPROVEMENT: Sector tags              │
│            │   extraction        │    (education, health, etc.)             │
│            │ • TextChunk objects │                                           │
│            └──────────┬──────────┘                                           │
│                       │                                                      │
│                       ▼                                                      │
│            ┌─────────────────────┐                                           │
│            │ processed_chunks.json│                                           │
│            │ (998 chunks stored)  │                                           │
│            └──────────┬──────────┘                                           │
│                       │                                                      │
└───────────────────────┼──────────────────────────────────────────────────────┘
                        │
┌───────────────────────┼──────────────────────────────────────────────────────┐
│                       ▼                                                      │
│                              LAYER 2: RETRIEVAL ENGINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│            ┌─────────────────────────────────────────┐                      │
│            │      PART B: RETRIEVAL ENGINE          │                      │
│            │                                          │                      │
│            │  ┌─────────────────────────────────┐    │                      │
│            │  │      EMBEDDING MODULE          │    │                      │
│            │  │                                 │    │                      │
│            │  │  • all-MiniLM-L6-v2 model      │    │                      │
│            │  │  • 384-dimensional vectors    │    │                      │
│            │  │  • Semantic encoding           │    │                      │
│            │  └────────────────┬────────────────┘    │                      │
│            │                   │                      │                      │
│            │  ┌────────────────▼────────────────┐    │                      │
│            │  │      VECTOR STORE              │    │                      │
│            │  │                                 │    │                      │
│            │  │  • ChromaDB PersistentClient  │    │                      │
│            │  │  • Cosine similarity search    │    │                      │
│            │  │  • Collection: 'acity_rag'     │    │                      │
│            │  └────────────────┬────────────────┘    │                      │
│            │                   │                      │                      │
│            │  ┌────────────────▼────────────────┐    │                      │
│            │  │    KEYWORD SEARCH (TF-IDF)   │    │                      │
│            │  │                                 │    │                      │
│            │  │  • Scikit-learn TF-IDF        │    │                      │
│            │  │  • 5000 max features          │    │                      │
│            │  │  • Keyword matching            │    │                      │
│            │  └────────────────┬────────────────┘    │                      │
│            │                   │                      │                      │
│            │  ┌────────────────▼────────────────┐    │                      │
│            │  │      HYBRID SEARCH            │◄───┼── IMPROVEMENT:       │
│            │  │                               │    │    k=10 (was k=3)    │
│            │  │  • 70% vector + 30% TF-IDF   │    │    3.3x more chunks  │
│            │  │  • Score aggregation          │    │                      │
│            │  │  • Metadata filtering         │◄───┼── IMPROVEMENT:       │
│            │  │    (sector-based)             │    │    Filter by tags    │
│            │  └────────────────┬────────────────┘    │                      │
│            │                   │                      │                      │
│            │  ┌────────────────▼────────────────┐    │                      │
│            │  │       RE-RANKER               │    │                      │
│            │  │                               │    │                      │
│            │  │  • Score normalization        │    │                      │
│            │  │  • Top-k selection            │    │                      │
│            │  │  • RetrievedDocument objects  │    │                      │
│            │  └────────────────┬────────────────┘    │                      │
│            │                   │                      │                      │
│            └───────────────────┼──────────────────────┘                      │
│                                │                                             │
│                                ▼                                             │
│                     Top-k Relevant Chunks                                    │
│                     (with scores & metadata)                                 │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────┼──────────────────────────────────────────────────────┐
│                       ▼                                                      │
│                           LAYER 3: PROMPT ENGINEERING                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│            ┌─────────────────────────────────────────┐                      │
│            │   PART C: PROMPT ENGINEERING           │                      │
│            │                                          │                      │
│            │  ┌─────────────────────────────────┐    │                      │
│            │  │   CONTEXT WINDOW MANAGER       │    │                      │
│            │  │                                 │    │                      │
│            │  │  • Token budget: 3000 tokens  │    │                      │
│            │  │  • Rank by relevance score      │    │                      │
│            │  │  • Truncate to fit              │    │                      │
│            │  └────────────────┬────────────────┘    │                      │
│            │                   │                      │                      │
│            │  ┌────────────────▼────────────────┐    │                      │
│            │  │      PROMPT BUILDER            │    │                      │
│            │  │                                 │    │                      │
│            │  │  Three Variants:               │    │                      │
│            │  │                                 │    │                      │
│            │  │  ┌─────────────┐  ┌──────────┐ │    │                      │
│            │  │  │   BASIC   │  │  STRICT  │ │    │                      │
│            │  │  │           │  │          │ │    │                      │
│            │  │  │ No rules  │  │5 explicit│ │    │                      │
│            │  │  │           │  │constraints│    │                      │
│            │  │  └─────────────┘  └──────────┘ │    │                      │
│            │  │         │              │         │    │                      │
│            │  │         └──────┬─────┘         │    │                      │
│            │  │                │               │    │                      │
│            │  │  ┌─────────────▼─────────────┐ │    │                      │
│            │  │  │      STRUCTURED          │ │    │                      │
│            │  │  │                           │ │    │                      │
│            │  │  │ • Bullet format required │ │    │                      │
│            │  │  │ • Source citations        │ │    │                      │
│            │  │  │ mandatory               │ │    │                      │
│            │  │  └───────────────────────────┘ │    │                      │
│            │  └────────────────┬────────────────┘    │                      │
│            │                   │                      │                      │
│            └───────────────────┼──────────────────────┘                      │
│                                │                                             │
│                                ▼                                             │
│                     Structured LLM Prompt                                    │
│                     (with anti-hallucination rules)                        │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                        │
┌───────────────────────┼──────────────────────────────────────────────────────┐
│                       ▼                                                      │
│                              LAYER 4: OUTPUT                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│            ┌─────────────────────────────────────────┐                      │
│            │      SIMULATED LLM RESPONSE            │                      │
│            │                                          │                      │
│            │  STRICT Prompt → Grounded response      │                      │
│            │  • "I don't know" when uncertain        │                      │
│            │  • Source citations [Source: X]          │                      │
│            │  • No hallucination                      │                      │
│            └─────────────────────────────────────────┘                      │
│                                                                               │
│            ┌─────────────────────────────────────────┐                      │
│            │      ADVERSARIAL TESTING               │                      │
│            │                                          │                      │
│            │  • Ambiguous query handling             │                      │
│            │  • False premise correction             │                      │
│            │  • RAG vs Pure LLM comparison          │                      │
│            └─────────────────────────────────────────┘                      │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Diagram

```
USER QUERY
    │
    ▼
┌────────────────────────────────────────┐
│ 1. QUERY PROCESSING                    │
│    • Parse & normalize                 │
│    • Log query metadata                │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│ 2. HYBRID RETRIEVAL                   │
│    │                                   │
│    ├─► Vector Search (Semantic)      │
│    │    • Embed query (384-dim)       │
│    │    • Similarity search           │
│    │                                  │
│    ├─► Keyword Search (TF-IDF)       │
│    │    • Term matching               │
│    │    • BM25 scoring                │
│    │                                  │
│    └─► Score Fusion (70/30 blend)     │
│         • Weighted combination         │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│ 3. METADATA FILTERING                  │
│    • Filter by sector tags            │
│    • Apply user constraints             │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│ 4. RE-RANKING & SELECTION             │
│    • Normalize scores                   │
│    • Select top-k (k=10)               │
│    • Deduplicate                        │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│ 5. CONTEXT WINDOW MANAGEMENT          │
│    • Rank by relevance                  │
│    • Truncate to 3000 tokens            │
│    • Select best subset                 │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│ 6. PROMPT CONSTRUCTION                 │
│    • Inject context                     │
│    • Add system instructions            │
│    • Apply style (Strict/Structured)   │
└────────────────────┬───────────────────┘
                     │
                     ▼
┌────────────────────────────────────────┐
│ 7. LLM RESPONSE                        │
│    • Generate answer                    │
│    • Cite sources                       │
│    • Admit uncertainty                  │
└────────────────────┬───────────────────┘
                     │
                     ▼
              FINAL RESPONSE
```

---

## Component Interactions

### 1. **Part A ↔ Part B Interaction**
```
┌──────────────┐      processed_chunks.json      ┌──────────────┐
│   PART A     │ ───────────────────────────────► │   PART B     │
│  (Data Prep) │                                  │ (Retrieval)  │
│              │      • TextChunk objects         │              │
│              │      • Source metadata           │              │
│              │      • Sector tags               │              │
└──────────────┘                                  └──────────────┘
```
**Flow**: Cleaned chunks with metadata → Indexed in vector store + keyword index

### 2. **Part B ↔ Part C Interaction**
```
┌──────────────┐      RetrievedDocument[]        ┌──────────────┐
│   PART B     │ ───────────────────────────────► │   PART C     │
│  (Retrieval) │                                  │   (Prompt)   │
│              │      • Text content              │              │
│              │      • Source attribution        │              │
│              │      • Relevance scores          │              │
└──────────────┘                                  └──────────────┘
```
**Flow**: Top-k chunks with scores → Context window → Prompt building

### 3. **Internal Component Dependencies**
```
┌──────────────────────────────────────────────────────────────┐
│                    RETRIEVAL ENGINE                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Embedding  │───►│ Vector Store │    │   Keyword   │     │
│  │   Module    │    │  (ChromaDB) │◄───│   Search    │     │
│  └─────────────┘    └──────┬──────┘    │  (TF-IDF)   │     │
│                            │           └─────────────┘     │
│                            │                │              │
│                            ▼                ▼              │
│                     ┌─────────────┐                     │
│                     │   Hybrid    │                     │
│                     │   Search    │                     │
│                     └──────┬──────┘                     │
│                            │                            │
│                            ▼                            │
│                     ┌─────────────┐                     │
│                     │  Re-ranker  │                     │
│                     │   (Top-k)   │                     │
│                     └─────────────┘                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Design Justification for Ghana Government Domain

### 1. **Why Hybrid Search (70% Vector + 30% TF-IDF)?**

| Aspect | Justification |
|--------|---------------|
| **Vector (70%)** | Captures semantic similarity for queries like "education spending" even if different terminology used ("school funding", "learning materials") |
| **TF-IDF (30%)** | Ensures exact keyword matches for critical terms like "GH₵564.6 million", "NPP", "2020" - essential for government data accuracy |
| **Blend** | Ghanaian government documents mix formal English with local terminology; hybrid approach catches both semantic meaning and exact figures |

**Domain Evidence**: Budget documents contain both narrative text ("allocations for education") and precise figures ("GH₵564.6 million"). Pure semantic search might miss exact numbers; pure keyword search misses context.

---

### 2. **Why Sentence-Aware Chunking?**

| Aspect | Justification |
|--------|---------------|
| **Government docs** | Parliamentary budget statements are structured as complete sentences: "Mr. Speaker, we have allocated..." |
| **Mid-sentence cuts** | Would break: "GH₵564.6 million for textbooks" → "564.6 million for" (meaningless) |
| **Semantic coherence** | Complete sentences preserve fiscal context essential for accurate retrieval |

**Domain Evidence**: Budget speeches follow formal parliamentary language. Breaking sentences destroys the "Mr. Speaker" context and numerical references.

---

### 3. **Why Strict Prompt Engineering?**

| Aspect | Justification |
|--------|---------------|
| **Government data** | High stakes - incorrect budget figures or election results have legal/political consequences |
| **Hallucination risk** | Pure LLM might "invent" allocations or vote counts not in documents |
| **"I don't know" rule** | Critical for: "What about defense spending?" when only education data retrieved |
| **Source citations** | Parliament requires traceability for all budget figures |

**Domain Evidence**: Government officials need verifiable sources for all claims. "I don't know" is preferable to confident misinformation in policy decisions.

---

### 4. **Why Metadata Filtering (Sector Tags)?**

| Aspect | Justification |
|--------|---------------|
| **Sector-specific queries** | Users ask: "What about health?" → System filters for health-tagged chunks |
| **Budget structure** | Ghana budget organized by sectors (Education, Health, Agriculture, etc.) |
| **Efficiency** | Reduces noise when user has clear domain interest |

**Domain Evidence**: Ministry of Finance budget documents are explicitly organized by MDAs (Ministries, Departments, Agencies). Sector tags mirror this structure.

---

### 5. **Why k=10 Retrieval (vs k=3)?**

| Aspect | Justification |
|--------|---------------|
| **Recall importance** | Government queries need comprehensive coverage, not just top-3 |
| **Context diversity** | Education budget spans textbooks, infrastructure, teacher salaries - need multiple chunks |
| **Reranking benefit** | More candidates → better chance of relevant context in final selection |

**Domain Evidence**: "Education allocation" appears in multiple budget sections: specific programs, sector summary, MDA breakdowns. k=10 captures this diversity.

---

### 6. **Why Adversarial Testing?**

| Aspect | Justification |
|--------|---------------|
| **Political sensitivity** | Users may ask leading questions ("Did NDC win?") with false premises |
| **Misinformation defense** | System must correct false claims with data, not amplify them |
| **Trust building** | Government systems must demonstrate reliability under adversarial conditions |

**Domain Evidence**: Election data is politically sensitive. System must handle biased queries factually, not reinforce misconceptions.

---

## Architecture Summary

| Component | Domain-Specific Design Choice | Rationale |
|-----------|------------------------------|-----------|
| **Data Prep** | Sentence-aware chunking + sector tags | Preserves parliamentary language, enables filtering |
| **Retrieval** | Hybrid (70/30) with k=10 | Balances semantic flexibility with exact figure matching |
| **Prompts** | Strict anti-hallucination rules | Government data requires verifiable accuracy |
| **Testing** | Adversarial queries | Political data demands misinformation resistance |

**Overall Assessment**: This architecture is specifically designed for **government document RAG** where **accuracy, traceability, and hallucination prevention** are non-negotiable requirements.
