# Main System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ACADEMIC CITY RAG SYSTEM                            │
│                    (Ghana Election & 2025 Budget RAG)                       │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌─────────────┐         ┌─────────────┐
     │   USER      │────────►│   QUERY     │
     │   INPUT     │         │   PARSER    │
     └─────────────┘         └──────┬──────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 1: DATA PREPARATION                         │
│                              (Part A)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐      │
│   │ Ghana Election  │      │  2025 Budget    │      │  Data Cleaner   │      │
│   │   CSV Data      │─────►│    PDF Data     │─────►│   & Chunker     │      │
│   │                 │      │                 │      │                 │      │
│   │ • Party votes   │      │ • Allocations   │      │ • Clean text    │      │
│   │ • Regional      │      │ • Initiatives   │      │ • Sentence      │      │
│   │   results       │      │ • Sectoral      │      │   chunks        │      │
│   └─────────────────┘      └─────────────────┘      └────────┬────────┘      │
│                                                            │                │
│                                                            ▼                │
│                                                   ┌─────────────────┐       │
│                                                   │ processed_chunks │       │
│                                                   │    .json         │       │
│                                                   │   (998 chunks)   │       │
│                                                   └────────┬────────┘       │
│                                                            │                │
└────────────────────────────────────────────────────────────┼────────────────┘
                                                             │
┌────────────────────────────────────────────────────────────┼────────────────┐
│                                                            ▼                │
│                          LAYER 2: RETRIEVAL ENGINE                          │
│                              (Part B)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────┐                            │
│                    │    RETRIEVAL ENGINE     │                            │
│                    │                         │                            │
│    ┌───────────────┼─────────────────────────┼───────────────┐             │
│    │               │                         │               │             │
│    ▼               │    ┌─────────────────┐  │               ▼             │
│ ┌──────┐          │    │   EMBEDDING     │  │          ┌──────┐          │
│ │Vector│◄──────────┼────│    MODULE       │  ├─────────►│TF-IDF│          │
│ │Store │          │    │                 │  │          │Search│          │
│ │      │          │    │ • all-MiniLM-L6-v2│  │          │      │          │
│ │Chroma│          │    │ • 384-dim vectors │  │          │ • BM25         │          │
│ │DB    │          │    │ • Semantic enc.   │  │          │ • Keywords     │          │
│ └──────┘          │    └────────┬────────┘  │          └──────┘          │
│    ▲               │             │          │               ▲             │
│    │               │             ▼          │               │             │
│    │               │    ┌─────────────────┐  │               │             │
│    │               │    │   HYBRID SEARCH │  │               │             │
│    │               │    │                 │  │               │             │
│    └───────────────┼────│ • 70% Vector     │◄─┘───────────────┘             │
│                    │    │ • 30% TF-IDF     │                                │
│                    │    │ • Score fusion   │                                │
│                    │    └────────┬────────┘                                │
│                    │             │                                         │
│                    │             ▼                                         │
│                    │    ┌─────────────────┐                                │
│                    │    │    RE-RANKER    │                                │
│                    │    │                 │                                │
│                    │    │ • Normalize     │                                │
│                    │    │ • Top-k = 10    │                                │
│                    │    │ • Deduplicate   │                                │
│                    │    └────────┬────────┘                                │
│                    │             │                                         │
│                    └─────────────┼─────────────────────────────────────────┘
│                                  │
│                                  ▼
│                       Top-k Relevant Chunks
│                       (with similarity scores)
│
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                                 ▼                                           │
│                       LAYER 3: PROMPT ENGINEERING                           │
│                              (Part C)                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────┐                            │
│                    │  PROMPT ENGINEERING     │                            │
│                    │                         │                            │
│                    │  ┌─────────────────┐    │                            │
│                    │  │ CONTEXT WINDOW  │    │                            │
│                    │  │    MANAGER      │    │                            │
│                    │  │                 │    │                            │
│                    │  │ • 3000 tokens   │    │                            │
│                    │  │ • Rank & select │    │                            │
│                    │  │ • Truncate      │    │                            │
│                    │  └────────┬────────┘    │                            │
│                    │           │              │                            │
│                    │           ▼              │                            │
│                    │  ┌─────────────────┐    │                            │
│                    │  │  PROMPT BUILDER │    │                            │
│                    │  │                 │    │                            │
│                    │  │ Three Styles:   │    │                            │
│                    │  │                 │    │                            │
│                    │  │ ┌──────┐┌──────┐│    │                            │
│                    │  ││BASIC ││STRICT││    │                            │
│                    │  ││      ││      ││    │                            │
│                    │  ││ None ││5rules││    │                            │
│                    │  │└──────┘└──────┘│    │                            │
│                    │  │     │     │     │    │                            │
│                    │  │  ┌──┴─────┐   │    │                            │
│                    │  │  │STRUCTURED│   │    │                            │
│                    │  │  │         │   │    │                            │
│                    │  │  │ Bullets │   │    │                            │
│                    │  │  │ Sources │   │    │                            │
│                    │  │  └─────────┘   │    │                            │
│                    │  └───────┬────────┘    │                            │
│                    │          │              │                            │
│                    └──────────┼──────────────┘                            │
│                               │                                             │
│                               ▼                                             │
│                    Structured LLM Prompt                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────────────┐
│                                 ▼                                           │
│                            LAYER 4: OUTPUT                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    ┌─────────────────────────┐                               │
│                    │    LLM RESPONSE         │                               │
│                    │    (Simulated)          │                               │
│                    │                         │                               │
│                    │ • Grounded answer      │                               │
│                    │ • Source citations       │                               │
│                    │ • "I don't know" if      │                               │
│                    │   uncertain              │                               │
│                    └───────────┬─────────────┘                               │
│                                │                                            │
│                                ▼                                            │
│                         Final Response                                     │
│                    (Delivered to User)                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Flow

```
┌──────────┐    JSON Chunks     ┌──────────┐
│  Part A  │ ─────────────────► │  Part B  │
│          │  (998 chunks)     │          │
└──────────┘                   │  Index   │
                               └────┬─────┘
                                    │
                         RetrievedDocument[]
                                    │
                                    ▼
┌──────────┐                   ┌──────────┐
│  Part C  │ ◄──────────────── │  Part B  │
│  Prompt  │    (Top-k docs)   │  Search  │
│  Build   │                   │          │
└────┬─────┘                   └──────────┘
     │
     │ LLM Prompt
     ▼
┌──────────┐
│ Response │
│  Output  │
└──────────┘
```

---

## Data Flow

```
User Query
    │
    ▼
┌─────────────────┐
│ 1. Parse Query  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Embed Query  │ (384-dim vector)
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌───────┐ ┌───────┐
│Vector │ │TF-IDF │
│Search │ │Search │
└───┬───┘ └───┬───┘
    │         │
    └────┬────┘
         │
         ▼
┌─────────────────┐
│ 3. Fuse Scores  │ (70% vector + 30% TF-IDF)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 4. Re-rank      │ (Top-k = 10)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 5. Select       │ (Context window: 3000 tokens)
│    Context      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 6. Build Prompt │ (Inject context + rules)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 7. Generate     │ (Simulated LLM response)
│    Response     │
└────────┬────────┘
         │
         ▼
   Final Output
```

---

## Core Components

| Layer | Component | Technology | Purpose |
|-------|-----------|------------|---------|
| **Data** | Chunker | NLTK + Custom | Sentence-aware chunking |
| **Data** | Metadata | Python dict | Sector tags (education, health) |
| **Retrieval** | Embeddings | all-MiniLM-L6-v2 | Semantic search (384-dim) |
| **Retrieval** | Vector DB | ChromaDB | Fast similarity search |
| **Retrieval** | Keywords | scikit-learn TF-IDF | Exact term matching |
| **Retrieval** | Hybrid | Score fusion | 70% vector + 30% TF-IDF |
| **Prompt** | Context Manager | Token counter | Fit to 3000 token limit |
| **Prompt** | Builder | String templates | 3 styles (Basic/Strict/Structured) |

---

## Key Files

| File | Layer | Purpose |
|------|-------|---------|
| `part_a_data_preparation.py` | Data | Chunking & cleaning |
| `modules/retrieval_engine.py` | Retrieval | Hybrid search orchestration |
| `modules/vector_store.py` | Retrieval | ChromaDB wrapper |
| `modules/keyword_search.py` | Retrieval | TF-IDF implementation |
| `part_c_prompt_engineering.py` | Prompt | Pipeline + experiments |
| `processed_chunks.json` | Data | Output of Part A |
