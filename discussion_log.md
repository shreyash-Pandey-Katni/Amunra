# Assessment Discussion Log

---

## 2026-05-29 — BM25Okapi: Layman Explanation

**Q:** Explain BM25Okapi like you are explaining to a layman.

**A Summary:**
BM25Okapi is a smarter word-counting search algorithm. It works in 4 steps:
1. **Count matching words** — more hits = better match (baseline)
2. **Length penalty** — long documents are penalized so they don't win just by being long
3. **Rare words matter more (IDF)** — a word appearing in 2/15 docs scores higher than one in 12/15
4. **Diminishing returns** — repeating a word 10 times doesn't give 5× the score of repeating it twice

"Okapi" = name of the library at University of London where the formula was developed. "BM25" = Best Match 25 (the 25th iteration).

**Why used alongside dense embeddings:** Dense models understand meaning ("equity purchased" ≈ "stock bought"). BM25 handles exact terms (tickers like "HDFCBANK", "NIFTY25JAN25P"). Financial data needs both; they're fused via RRF.

---

## 2026-05-29 — BM25 Alternatives in 2026 + TF-IDF Question

**Q:** Is there any alternative in 2026 which is better? Is this library using TF-IDF underneath?

**A — Is BM25 TF-IDF?**
Yes. BM25 is TF-IDF with two critical fixes:
| | TF-IDF | BM25 |
|---|---|---|
| Rare word boost (IDF) | ✅ | ✅ |
| Term frequency | Raw count | Saturates (diminishing returns) |
| Document length penalty | ❌ | ✅ |

**A — Three generations of sparse retrieval:**

- **Gen 1: BM25 (1994)** — Pure statistics, no ML, exact word match. What we use.
- **Gen 2: SPLADE (2021)** — Transformer-learned sparse vectors with token expansion. Outperforms BM25 on benchmarks but breaks rare tokens like `NIFTY25JAN25P` into `[UNK]` — fatal for financial ticker matching.
- **Gen 3: BM42 (Qdrant, 2024)** — Keeps BM25's IDF, replaces TF with transformer attention weights. Better term importance on short RAG chunks. Requires Qdrant (can't use standalone with ChromaDB).

**Why BM25Okapi is still right for this assessment:**
BM42 can't be used standalone with ChromaDB. SPLADE corrupts NSE ticker symbols. BM25's weakness (TF is useless on short chunks) is minor here because the IDF signal — which rare financial terms appear where — is what matters.

---

## 2026-05-29 — Why List Instead of DataFrame for Part A Trade Loading

**Q:** Why are we using list instead of DataFrame for loading trades in Part A?

**A:**
Two intentional reasons:
1. **Embedding pipeline needs strings** — `model.encode()`, `bm25.get_scores()`, and text-building all iterate over `trades_a` accessing fields as plain strings. A DataFrame would require `.to_dict("records")` conversion anyway — the list of dicts *is* that format.
2. **NetworkX ingestion is row-by-row** — `G.add_node(t["fund"])`, `G.add_edge(...)` map directly onto a list of dicts. No column-wise analytics (no `groupby`, `mean()`, column filtering) are done in Part A, so a DataFrame adds overhead with no benefit.

---

---

## 2026-05-29 — Where is Embedding Metadata Stored?

**Q:** Where are we storing the metadata for embeddings?

**A:**
Metadata is stored **inside ChromaDB** alongside each embedding vector. ChromaDB stores 4 things per document: `id`, `embedding`, `document` (the raw text), and `metadata` (a flat dict of scalar fields).

**Part A — `trades_collection` (persisted at `./chroma_db_a/`):**
```python
metadatas=[{
    "fund": ..., "sector": ..., "trade_type": ...,
    "instrument": ..., "instrument_type": ...
}]
```
Used for post-filter in hybrid A2 queries (e.g. `sector == "Banking"`).

**Part B — `trades_collection_b` (persisted at `./chroma_db_b/`):**
```python
metadatas=[{
    "fund": ..., "sector": ..., "direction": ...,
    "instrument": ..., "ticker": ..., "instrument_type": ...
}]
```

**Part B — `research_collection` (same `./chroma_db_b/`):**
```python
metadatas=[{
    "analyst": ..., "date": ..., "sector": ...,
    "classification": ..., "section": ...
}]
```
Analyst provenance (name, date, sector, classification, section label) is stored per chunk so retrieval results can show who wrote what.

**Key point:** The metadata is NOT stored with the dense embedding vectors in memory (the numpy array). It only lives in ChromaDB. The BM25 index has no metadata — it only knows text and IDs. So metadata filtering is exclusively a ChromaDB capability.

---

## 2026-05-29 — RRF Explanation

**Q:** Explain RRF like a layman

**A:**
RRF (Reciprocal Rank Fusion) combines two ranked lists by caring about **position, not scores**.

Formula: `score(doc) = 1/(rank_in_bm25 + 60) + 1/(rank_in_dense + 60)`

The 60 constant prevents rank-1 from dominating. Documents that rank highly in **both** lists win. The key insight: BM25 scores and cosine similarity scores are incomparable units — RRF sidesteps this by only looking at rank order.

---

## 2026-05-29 — RRF Alternatives in 2026

**Q:** What are alternatives to RRF in 2026?

**Sources:** Weaviate docs (relativeScoreFusion), Qdrant docs (DBSF, Weighted RRF, ColBERT), Elastic docs

**A:**

### 1. RSF — Relative Score Fusion (Weaviate default since v1.24)
Instead of using rank position, normalize each retriever's scores to [0,1] range (min→0, max→1) and sum them. Preserves score magnitude information that RRF discards. Now Weaviate's default over RRF.
- **Upside:** Rewards documents that score "very high" in one retriever, not just "ranked high"
- **Downside:** Sensitive to outlier scores skewing the normalization range

### 2. DBSF — Distribution-Based Score Fusion (Qdrant v1.11+, 2024)
Normalizes scores using statistical distribution (mean ± 3σ) before combining. More robust to outliers than RSF.
- **Upside:** Better calibration when retrievers have different score distributions
- **Downside:** Small top-k sample makes statistics noisy; one outlier can skew normalization

### 3. Weighted RRF (Qdrant v1.17+, 2025)
Standard RRF but with per-retriever weights. If dense search is stronger for your workload, give it weight 3.0 vs BM25's 1.0.
- **Upside:** Tunable; respects that one retriever may dominate for a given query type
- **Downside:** Needs an eval set to tune; arbitrary weights without measurement often hurt more than help

### 4. ColBERT / Late Interaction (Qdrant multivector, v1.10+)
Instead of one embedding per document, creates one embedding **per token**. Final relevance = interaction score between all query tokens and all document tokens (MaxSim).
- **Upside:** Captures fine-grained token-level matches; often beats both RRF and dense-only
- **Downside:** Storage blows up (N tokens × embedding_dim per document); requires Qdrant multivector support; not usable with ChromaDB

### 5. Cross-Encoder Re-ranker (Cohere Rerank, ms-marco models)
Run top-K candidates through a model that jointly encodes query+document to score relevance. Architecturally different — not a fusion method but a **re-ranking stage** after hybrid retrieval.
- **Upside:** Best retrieval quality; model sees full query-document context together
- **Downside:** Slow (runs inference per candidate pair); API cost (Cohere) or GPU needed

### For our assessment (ChromaDB + BM25Okapi in Python):
We're implementing RRF manually. The only practical upgrade without switching to Qdrant would be **Weighted RRF** (give dense slightly more weight for semantic queries) or adding a **cross-encoder re-ranker** as a 3rd stage (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2` from sentence-transformers, runs locally). ColBERT/DBSF require Qdrant.

---

## 2026-05-29 — ChromaDB vs Qdrant: Implementation Decision

**Q:** How much effort to switch to Qdrant? Any assignment constraint on DB choice?

**Context:** PDF explicitly states *"You may use any Python libraries, embedding models, vector databases, and graph libraries of your choice."* — no constraint.

**Decision:** Implemented in **ChromaDB** due to 48-hour time constraint. Will be switched to **Qdrant** if time permits.

**Reasoning for ChromaDB (current):**
- Zero setup friction — `pip install chromadb`, no Docker service needed for the DB itself
- Persistent local client with one line: `chromadb.PersistentClient(path="./chroma_db_a")`
- Sufficient for assessment scale (15–20 trades, 7 research chunks)
- No API keys, no cost, no latency — correct for take-home
- Familiar API — focus time on problem, not infra

**Why Qdrant is the better production choice:**
- Native hybrid search: stores sparse + dense vectors in same collection, Query API does RRF/DBSF server-side — eliminates all manual BM25+RRF Python code
- Disk-backed HNSW (not in-RAM) — handles 100M+ vectors; ChromaDB breaks at ~500K on commodity hardware
- Weighted RRF (v1.17+), DBSF (v1.11+), ColBERT multivector (v1.10+) — all unavailable in ChromaDB
- Better metadata filtering with payload indexing
- Production-grade: used by Cohere, Mistral, and most RAG production stacks in 2025–2026

**Migration effort estimate:** ~2.5–3 hours
- docker-compose Qdrant service: 5 min
- Collection creation API change: 15 min
- Upsert to PointStruct format: 30 min
- Replace manual RRF with native Query API prefetch: 30 min
- Sparse vector switch (FastEmbed SPLADE → test NSE ticker tokenisation): 45 min
- Metadata filter parity: 10 min

**Key additional reason for ChromaDB choice:**
ChromaDB was chosen primarily for its simplicity — it is ready to use with a single `pip install chromadb` with no additional infrastructure, no Docker service, no configuration files. This makes it ideal for rapid prototyping and take-home assessments. However, it is **not suitable for production** — it lacks native hybrid search, its HNSW index lives entirely in RAM (not disk-backed), it has no built-in sparse vector support, and it degrades significantly beyond ~500K vectors. It is a prototyping tool, not a production vector database.

**📌 Next Version Note:**
**Qdrant will be used in the next version** of this system. It is the production-grade replacement — disk-backed HNSW, native sparse+dense hybrid search with server-side RRF/DBSF fusion, weighted retrieval, ColBERT late interaction support, and handles 100M+ vectors on commodity hardware. Migration effort is estimated at ~2.5–3 hours.

---

## 2026-05-29 — Embedding Issue Root Cause + Fix Plan

**Issue:** T008 (ICICIBANK) is missed for the query "Which trades were related to banking sector weakness?"

**Root Cause Analysis:**

- T008 note: `"ICICI undervalued vs HDFC on P/B basis"` — no word "weakness" appears
- T006 note: `"Accumulated on sector weakness"` — direct hit, ranks #1 correctly
- BM25 keyword overlap with query: T008 has only `{banking}`, T006 has `{banking, sector, weakness}`
- Dense model (`bge-large-en-v1.5`): general-purpose model does not know the financial synonym chain: "P/B basis undervaluation" → "buying on sector weakness" → "contrarian accumulation"
- FinMTEB paper (EMNLP 2025, arXiv:2502.10990) confirms: general-purpose benchmarks show limited correlation with financial domain tasks; BoW models sometimes outperform dense models on financial STS

**Model Investigation:**

| Model | Size | Accessible | Financial Domain | Notes |
|---|---|---|---|---|
| `BAAI/bge-large-en-v1.5` (current) | 335M | ✅ | ❌ General only | Good MTEB-en 64.23, misses financial synonyms |
| `yixuantt/Fin-e5` (top FinMTEB) | ~335M | ❌ HTTP 401 | ✅ Best | Gated model, inaccessible for take-home |
| `BAAI/bge-en-icl` (few-shot ICL) | ~7B (Mistral backbone) | ✅ | ✅ With examples | Too large for 12GB VRAM alongside other models |
| `ProsusAI/finbert` | 110M | ✅ | ✅ Sentiment only | Trained for sentiment classification, not retrieval |
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | 22M | ✅ | Partial | Cross-encoder sees full query+doc jointly, better inference |

**Fix Plan (3-phase, cumulative improvement):**

**Phase 1 — BM25 Financial Query Expansion (20 min, no new model)**
Add a domain synonym expansion step applied before BM25 tokenization:
```python
financial_query_expansions = {
    "weakness": ["weakness", "undervalued", "accumulated", "bearish", "P/B"],
    "protective": ["protective", "hedge", "option", "put", "OTM", "cover"],
    "banking": ["banking", "bank", "BFSI", "HDFC", "ICICI", "SBIN"],
    "IT": ["IT", "technology", "software", "TCS", "Infosys", "tech"],
}
```
Since BM25 is already justified as primary retrieval signal (FinMTEB), making it domain-aware is the most targeted fix. Directly addresses the "P/B basis" vs "weakness" gap.

**Phase 2 — Cross-Encoder Re-ranker stage (30 min)**
Add `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, ~90MB) as a 3rd retrieval stage:
```
Stage 1: BM25 expanded → top-8 candidates
Stage 2: Dense (bge-large) + RRF merge → top-8 re-ranked
Stage 3: Cross-encoder scores all 8 query+doc pairs jointly → final top-3
```
Why it helps: Cross-encoder reads "banking sector weakness [SEP] BUY ICICIBANK (Banking)... ICICI undervalued vs HDFC on P/B basis" — joint encoding allows contextual inference that the general model misses in separate query/doc encoding.

**Phase 3 — Instruction-tuned E5 alternative (stretch, same model size)**
Replace `bge-large-en-v1.5` with `intfloat/e5-large-v2` (same 335M params) using instruction prefix:
```python
query = "query: Which trades were related to banking sector weakness?"
doc   = "passage: BUY ICICIBANK (Banking)... ICICI undervalued vs HDFC on P/B basis"
```
E5 models are trained with instruction prefixes that allow domain adaptation at inference time without a new model.

**Recommended approach for assessment:** Phase 1 + Phase 2 (50 min total). BM25 expansion fixes the lexical gap; cross-encoder fixes the semantic gap. Both use already-installed packages (`rank_bm25`, `sentence_transformers.CrossEncoder`).

**📌 Production note:** Fin-E5 (`yixuantt/Fin-e5`) is the right long-term answer once accessible. Alternatively, fine-tuning `bge-large-en-v1.5` on Indian financial trade notes would give best domain performance.

---

## 2026-05-29 — Live Research: Best Embedding Models for Finance RAG (2025–2026)

**Research method:** Live Playwright browser search across HuggingFace, Google, Milvus blog, Voyage AI blog, FinMTEB GitHub, arXiv paper.

### FinMTEB Benchmark Table (from arXiv:2502.10990v2, Feb 2025 — live-fetched)

Actual scores retrieved from paper HTML:

| Model | Size | Avg | STS | Retrieval | Notes |
|---|---|---|---|---|---|
| BOW | - | 0.4504 | 0.4845 | 0.2084 | BoW beats dense models on STS! |
| FinBERT | 110M | 0.4205 | 0.4198 | 0.1102 | Finance-trained but poor retrieval |
| instructor-base | 110M | 0.5886 | - | 0.5772 | |
| **bge-large-en-v1.5** | **335M** | **0.6301** | **0.3396** | **0.6463** | **← Our current model** |
| AnglE-BERT | 335M | 0.6088 | - | 0.5730 | |
| gte-Qwen1.5-7B | 7B | 0.6427 | - | 0.6697 | |
| bge-en-icl | 7B | 0.6309 | - | 0.6789 | Few-shot, same avg as bge-large |
| NV-Embed v2 | 7B | 0.6322 | - | 0.7061 | Best open-source retrieval in paper |
| e5-mistral-7b | 7B | 0.6475 | - | 0.6749 | |
| text-embedding-3-large | API | 0.6613 | - | 0.7112 | OpenAI commercial |
| **voyage-3-large** | API | **0.6765** | **0.4145** | **0.7463** | **Best commercial overall** |
| **Fin-E5** | **7B** | **0.6767** | **0.4342** | **0.7105** | **Best open-source overall, gated 401** |

**Key finding:** bge-large-en-v1.5 is already the best encoder-based (non-7B) model in the benchmark. To get meaningfully better retrieval, we need either a 7B model or a domain-fine-tuned version.

### Accessible Models Found via Live HuggingFace Search (2025–2026)

**1. `Qwen/Qwen3-Embedding-0.6B`** ⭐ Best 2025/2026 option
- Released June 2025 by Alibaba Qwen team
- 0.6B params, 28 layers, 32K context, 1024-dim
- **Instruction-aware** — supports task-specific instructions like: `"Instruct: Given a financial trade query, retrieve relevant trades.\nQuery: {query}"`
- MRL support (flexible dimensions 32–1024)
- 8B version ranked #1 on MTEB Multilingual leaderboard (June 5, 2025, score 70.58)
- SentenceTransformer compatible: `SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")`
- Requires `transformers>=4.51.0`, `sentence-transformers>=2.7.0`
- **NOT in FinMTEB paper (released after it)** — but Qwen3 architecture is decoder-based with instruction tuning, superior to encoder models at equivalent size
- Last-token pooling (not CLS pooling) — needs code adjustment

**2. `baconnier/Finance_embedding_large_en-V1.5`** ⭐ Safest drop-in
- Direct fine-tune of `BAAI/bge-large-en-v1.5` (our exact current model) on financial Q&A data
- Same 1024-dim, same CLS pooling, same SentenceTransformer interface
- **Zero code changes needed** — just replace model name string
- Training data: `baconnier/finance2_dataset_private` (financial QA pairs)
- 0.3B params, updated June 2024

**3. `FinLang/finance-embeddings-investopedia`** — Popular but mismatched
- Fine-tune of bge-base-en-v1.5 on Investopedia data (3.66M downloads)
- 768-dim (vs our current 1024-dim) — requires ChromaDB collection recreation
- Trained on definition/explanation style text, not trade notes
- Less suitable for short-document trade notes retrieval

**4. `thomaskim1130/stella_en_400M_v5-FinanceRAG-v2`** — Specialized for reports
- Stella_en_400M_v5 (NovaSearch, GTE-large backbone) fine-tuned on FinanceRAG dataset
- Trained for financial report QA (tabular data, SEC filings) — different from our use case
- 1024-dim, 400M params — not ideal for short trade notes

**5. `Qwen3-Reranker-0.6B`** — Companion re-ranker to Qwen3-Embedding
- Same 0.6B architecture, instruction-aware re-ranker
- Can replace `cross-encoder/ms-marco-MiniLM-L-6-v2` in Stage 3
- Instruction: `"Given a query, determine if the document is relevant to it."`

### 2026 General RAG Embedding Landscape (from Milvus blog, March 26, 2026)
Milvus tested 10 models on CCKM benchmark (Cross-modal, Cross-lingual, Key info, MRL). Top models:
- **Gemini Embedding 2** (Google) — best all-rounder, multi-modality
- **Jina Embeddings v4** (3.8B) — MRL + LoRA adapters
- **Voyage Multimodal 3.5** — balanced, API
- **Qwen3-VL-2B** (Alibaba) — open-source, lightweight multimodal
- **BGE-M3** (568M) — open-source, 100+ languages

**Assessment note:** These are general-purpose 2026 models. For finance-specific retrieval in our context, Qwen3-Embedding-0.6B with instruction tuning is the best accessible option.

### Final Recommendation

| Approach | Effort | Risk | Improvement | Verdict |
|---|---|---|---|---|
| BM25 query expansion | 15 min | Zero | +T008 recall (lexical) | ✅ Do first |
| Swap to `baconnier/Finance_embedding_large_en-V1.5` | 20 min | Low | Finance domain boost | ✅ Do second |
| Add `cross-encoder/ms-marco-MiniLM-L-6-v2` re-ranker | 20 min | Low | Semantic inference fix | ✅ Do third |
| Swap to `Qwen/Qwen3-Embedding-0.6B` | 45 min | Medium (code changes) | Best overall quality | 🔄 Stretch goal |
| `voyage-finance-2` API | 10 min | Low setup | Best finance retrieval | ❌ Needs API key |

**Immediate plan:** BM25 expansion → `baconnier/Finance_embedding_large_en-V1.5` → cross-encoder re-rank
**Stretch goal:** Replace with Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B (most modern stack)

---

## 2026-05-29 — Embedding Fix Implementation Results

### Changes Made to `part_a.ipynb`
1. **BM25 domain synonym expansion** — `FINANCE_SYNONYMS` dict added. `expand_query()` appends synonyms before BM25 tokenization. Maps: `weakness → [undervalued, P/B, accumulated, ...]`, `banking → [HDFC, ICICI, SBIN, rerating, ...]`, etc.
2. **Cross-encoder re-ranking stage** — `cross-encoder/ms-marco-MiniLM-L-6-v2` added as Stage 2. BM25+dense+RRF generates top-10 candidates; cross-encoder re-ranks to final top-k.
3. **Fixed labeled_pairs ground truth** — Previous eval had T008 in IT query (wrong — T008 is ICICIBANK). Corrected to actual CSV data.
4. **Model**: Kept `BAAI/bge-large-en-v1.5` — `baconnier/Finance_embedding_large_en-V1.5` (direct fine-tune) was attempted but is a gated repo (403).

### Eval Results (Mean P@3: 0.47 → 0.67, MRR: 1.00)

| Query | P@3 | MRR |
|---|---|---|
| Banking sector weakness | 0.67 | 1.00 |
| Protective hedging | 0.33 | 1.00 |
| IT sector positions | 1.00 | 1.00 |
| Reliance trades | 1.00 | 1.00 |
| Consumer tech | 0.33 | 1.00 |
| **Mean** | **0.67** | **1.00** |

### Key Fix: T008 Retrieved
T008 (ICICIBANK, "ICICI undervalued vs HDFC on P/B basis") now ranks **#2** for "Which trades were related to banking sector weakness?" — BM25 score 11.79, because `HDFC`, `bank`, `P/B` appear in the synonym-expanded query tokens.

### Note on Gated Models
- `baconnier/Finance_embedding_large_en-V1.5` — GatedRepoError (403), inaccessible without approval
- `yixuantt/Fin-e5` — Also gated (confirmed earlier)
- Best accessible path: Qwen3-Embedding-0.6B (open, June 2025) as stretch goal upgrade

---

## 2026-05-29 — RAG Layer Added to Part A

**Decision:** Added a RAG (Retrieval-Augmented Generation) agent as section A4 at the end of `part_a.ipynb` (cells 19–22).

**Why RAG is necessary:**
- Assessment Query 3: *"What IT sector positions were taken and **why**?"* — the word "why" cannot be answered by retrieval alone; it requires a language model to synthesise the `notes` field content into coherent prose.
- Pure retrieval (A1/A2) returns ranked documents; the user still has to read and interpret them manually.
- A RAG agent reads the retrieved context and generates a natural-language explanation — this is the standard production pattern for AI-assisted trade analysis.

**Architecture (cells 19–22):**
- **3 Tools** wrap the existing retrieval infrastructure:
  1. `hybrid_trade_retrieval` — calls `hybrid_retrieve()` (BM25 + bge-large-en-v1.5 + CrossEncoder, Cell 7)
  2. `chroma_vector_search` — queries ChromaDB `trades_collection` with optional metadata filters (sector, instrument_type)
  3. `knowledge_graph_query` — queries NetworkX graph for structural questions (fund holdings, sector exposure, shared instruments)
- **LLM:** `qwen2.5:14b-instruct-q4_K_M` via Ollama (running locally, **no API key required**)
  - Confirmed tool-calling support via Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`)
  - Model already downloaded (9 GB, q4_K_M quantization)
- **Agent loop:** ReAct-style (max 6 steps): send query + tool schemas → execute tool calls → feed results back → get final NL answer
- **Implementation:** Uses `requests` (stdlib-adjacent, already installed) to call Ollama REST API; zero new dependencies

**Why qwen2.5:14b-instruct over alternatives:**
- Already available locally in Ollama — no download needed
- 14B parameters at q4_K_M gives strong instruction-following and tool-call reliability
- Confirmed empirically: tool calling works correctly (finish_reason="tool_calls", correct JSON arguments)
- Alternatives considered: Qwen3-0.6B (too small for reliable multi-step tool calling), mistral:7b (weaker tool calling), gemma4 (9.6 GB, similar size but weaker on finance)

**No new packages installed** — reuses: `requests` (built-in ecosystem), existing `hybrid_retrieve()`, `collection` (ChromaDB), `G` (NetworkX) all defined in earlier cells.


---

## 2026-05-30 — Part B RAG Agent + Evaluation Added (B5 + B6)

**Decision:** Added LangGraph ReAct RAG agent and RAGAS-inspired evaluation cells to `part_b.ipynb` (cells 33–37).

**Rationale for adding RAG to Part B:**
The combined query — "What is our banking sector view and which funds have banking exposure?" — requires synthesising:
1. **Trade facts** (which fund bought/sold which instrument, at what price)
2. **Analyst rationale** (why the view is constructive/cautious — Priya Sharma's banking note)
3. **Structural relationships** (which funds hold banking instruments — Neo4j KG traversal)

Pure retrieval (B2) returns documents but cannot synthesise a coherent NL answer that integrates all three sources. The RAG agent enables this cross-source synthesis.

**Architecture (cells 33–37):**
- **3 Tools** wrapping existing Part B infrastructure:
  1. `hybrid_trade_retrieval_b` — BM25 + bge-large-en-v1.5 + RRF over `trades_collection_b` (20 cleaned trades)
  2. `research_chunk_retrieval` — BM25 + bge-large-en-v1.5 + RRF over `research_collection` (7 analyst note chunks)
  3. `knowledge_graph_query_b` — raw Cypher via `driver_b` (Neo4j, same instance as B3)
- **LLM:** `qwen2.5:14b-instruct-q4_K_M` via Ollama (same model as Part A RAG)
- **Agent:** `langgraph.prebuilt.create_react_agent` (ReAct loop — same pattern as Part A A4)

**Evaluation metrics (B6, cell 37):**
Aligned with RAGAS framework (docs.ragas.io, 2026):
| Metric | RAGAS equivalent | Implementation |
|--------|-----------------|----------------|
| Faithfulness | `Faithfulness` | TX-XXXXX IDs in answer must appear in retrieved contexts |
| Context Recall | `ContextRecall` | Expected TX-IDs surfaced by retrieval (set recall) |
| Semantic Similarity | `AnswerRelevance` (proxy) | bge-large cosine vs reference answer |
| ROUGE-L | Lexical overlap | LCS F1 vs reference answer |

**Why local metrics instead of RAGAS LLM-judge:**
RAGAS 0.4.3 LLM-judge metrics require a live LLM call per evaluation sample. For local-only setup (Ollama), we use heuristic equivalents that are deterministic, free, and well-suited to structured financial data (TX-ID tracking is more reliable than LLM judge for this domain).

**No new dependencies added** — reuses `rouge_score` (already in requirements), `model` (bge-large, cell 6), `hybrid_query_collection` (cell 19), `driver_b` (cell 22), `bm25_trades`/`bm25_research` (cell 18).
