# Finsight — Financial Intelligence & Research Agent

> **An evidence-first financial research system that turns filings, earnings material, structured financial data, and market news into traceable answers — while making retrieval quality measurable instead of hiding it behind an LLM.**

Finsight is a Bloomberg-style financial intelligence project designed to help an analyst move from a natural-language question to **relevant evidence, structured financial context, and an explainable answer**.

The core idea is simple:

**Question → Intent understanding → Hybrid retrieval → Candidate filtering → Reranking → Evidence selection → LLM synthesis → Source verification**

Instead of asking a language model to "know" a company's financial story, Finsight attempts to retrieve the underlying evidence first and then uses the LLM as a synthesis layer.

---

## Why this project matters

Financial research is not just a question-answering problem.

A real analyst may ask:

- "What did management say about the future outlook?"
- "What was JPMorgan's revenue in 2024?"
- "How has revenue changed across quarters?"
- "What are the major risks facing the company?"
- "What did management say during the earnings call?"
- "Why did profitability change?"
- "What are the company's major capital-allocation priorities?"

These questions require **different kinds of evidence**.

A single vector search strategy can easily return documents that are semantically related but analytically wrong. A proxy statement may mention strategy, a risk filing may mention "future," and an MD&A section may contain the word "outlook" — but that does not mean any of them answers the analyst's actual question.

Finsight therefore treats retrieval as an engineering problem:

> **The goal is not to retrieve text that looks similar to the question. The goal is to retrieve evidence that can actually support the answer.**

---

# Real-world use cases

## 1. Equity research

An analyst researching JPMorgan, Apple, Meta, or another public company can ask financial questions in natural language instead of manually searching hundreds of pages of filings.

Example:

> **"What did JPMorgan's management say about its future outlook?"**

The system attempts to prioritize management commentary, earnings material, guidance, MD&A, prepared remarks, and related evidence rather than blindly returning the nearest semantic match.

---

## 2. Fundamental financial analysis

For questions such as:

> **"What was Meta's revenue in 2024?"**

the system can prioritize structured financial data, financial tables, XBRL-derived information, computed metrics, and relevant filing content.

This is especially important for numeric questions because a narrative paragraph that happens to mention revenue is not necessarily the best source for the actual reported figure.

---

## 3. Earnings-call research

Management commentary is often distributed across prepared remarks, Q&A, earnings transcripts, presentations, and filings.

Finsight is designed to distinguish these evidence types and retrieve the material most appropriate to the question.

Example:

> **"What concerns did management discuss about demand?"**

This is fundamentally different from:

> **"What was revenue?"**

The retrieval strategy should change accordingly.

---

## 4. Trend and quarterly analysis

Financial questions frequently require evidence from multiple periods.

Examples:

- "How did JPMorgan's revenue change over the last four quarters?"
- "Is operating margin improving?"
- "What is the revenue trend?"
- "Compare quarterly earnings."

For these tasks, diversity is important because retrieving five nearly identical chunks from one quarter is not useful.

Finsight therefore treats trend-style retrieval differently from ordinary semantic retrieval.

---

## 5. Investment research workflows

The project can act as a research layer between raw financial information and an analyst.

Instead of manually jumping between:

- SEC filings
- earnings material
- financial tables
- structured metrics
- company disclosures
- financial news

the user can ask a question and receive a synthesized response with supporting source references.

This does **not** replace an analyst.

It reduces the mechanical work involved in finding and organizing evidence.

---

# Architecture

```text
                         ┌─────────────────────┐
                         │     User Query      │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Query Classification│
                         │ + Intent Analysis   │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
        ┌──────────────────┐                ┌──────────────────┐
        │ Semantic Search  │                │      BM25        │
        │   / pgvector     │                │ Keyword Retrieval│
        └────────┬─────────┘                └────────┬─────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │ Hybrid Score        │
                         │ + Quality + Section │
                         │ + Source-Aware      │
                         │   Boosting          │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Candidate Pool      │
                         │ + Diversity Control │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Cross-Encoder       │
                         │ Reranking           │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Final Evidence      │
                         │ Selection           │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Qwen LLM            │
                         │ Evidence Synthesis  │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Answer + Sources    │
                         │ + Verification      │
                         └─────────────────────┘
```

---

# Core capabilities

### Hybrid retrieval

Finsight combines:

- **Semantic/vector retrieval** for meaning
- **BM25** for exact financial terminology and keywords
- **Synonym/query expansion** to improve lexical recall
- **Section-aware ranking**
- **Source-type-aware ranking**
- **Quality weighting**
- **Cross-encoder reranking**

This is designed to reduce the weaknesses of relying on any single retrieval method.

---

### Query-aware retrieval

The query is classified into a dominant analytical task such as:

- financial metric
- growth
- trend analysis
- quarterly comparison
- balance sheet
- capital allocation
- risk
- management commentary
- segment analysis
- news sentiment
- price
- comparison

The classifier also produces signals such as whether the query needs:

- computed metrics
- transcript evidence
- trend evidence
- preferred sections
- excluded sections

The retrieval pipeline then adapts to the task.

---

### Section-aware evidence selection

Financial documents contain very different types of information.

Finsight can distinguish sections such as:

- MD&A
- earnings
- financial tables
- income statement
- risk factors
- guidance
- prepared remarks
- Q&A
- forward-looking
- segment
- computed metrics
- structured metrics
- XBRL-related data

This lets the retrieval system consider **where information comes from**, not only what words it contains.

---

### Pre-rerank diversity

A major retrieval failure mode is redundancy.

If the candidate pool contains many near-identical chunks, a reranker can spend its entire budget scoring variations of the same evidence.

Finsight therefore includes a pre-rerank diversity stage that removes overly similar candidates before reranking.

For trend queries, diversity filtering can be disabled because multiple similar-looking quarterly chunks may actually represent different periods and therefore different evidence.

---

### Cross-encoder reranking

Initial retrieval is optimized for recall.

Reranking is used to answer a harder question:

> "Given the actual query and this candidate chunk, how relevant is this chunk?"

The cross-encoder produces a relevance score that is combined with retrieval signals and task-specific adjustments.

---

### Evidence-aware LLM generation

The LLM is used primarily as a **reasoning and synthesis layer over retrieved evidence**.

The system passes selected chunks to the model rather than expecting the model to generate the financial answer from memory.

This makes the architecture closer to:

**Retrieval → Evidence → Synthesis**

rather than:

**Question → Hallucinated answer**

---

# Data and technology stack

The project currently uses components including:

- **Python**
- **Qwen 2.5 3B Instruct**
- **Ollama**
- **BGE embeddings**
- **Cross-encoder reranking**
- **BM25**
- **pgvector / Supabase**
- **FAISS** for retrieval experimentation
- **SEC EDGAR filings**
- **Yahoo Finance news**
- Structured and computed financial data
- **FastAPI** planned/used as the serving layer
- **Docker** for deployment
- **AWS** as the deployment target

The exact production configuration may evolve as the retrieval and evaluation layers mature.

---

# Retrieval scoring philosophy

Finsight does not treat one score as the entire truth.

A candidate can receive signals from multiple stages:

```text
Semantic relevance
        +
BM25 relevance
        +
Quality
        +
Section preference
        +
Source-type preference
        +
Task-specific weighting
        +
Cross-encoder relevance
        =
Final retrieval priority
```

This makes the ranking pipeline more controllable and debuggable.

The project also exposes retrieval diagnostics so failures can be inspected instead of hidden.

---

# Observability and debugging

One of the most important aspects of Finsight is that retrieval is intentionally observable.

A typical diagnostic output exposes information such as:

```text
rerank=0.9572
final=1.7075
section=computed_metrics
type=computed_metrics
hybrid=0.8300
```

and:

```text
chunks to llm: 5
```

This allows the developer to inspect:

- which chunks were retrieved
- their hybrid score
- reranker score
- final score
- section
- source type
- document date
- exact text passed to the LLM

That matters because an incorrect answer is not always an LLM problem.

It may be:

**query classification → retrieval → ranking → evidence selection → synthesis**

The diagnostics make it possible to determine where the failure occurred.

---

# A real failure the system exposed

During testing with a question similar to:

> **"What did Apple's management say about its future outlook?"**

the retrieval diagnostics showed a serious issue.

The highest-ranked evidence included:

- an Apple 2024 10-K MD&A chunk that primarily contained stock-repurchase and document-navigation material
- a 2026 DEF 14A executive compensation section
- a general 10-K section
- a generic forward-looking-statements section
- other MD&A material

The problem was not simply that the system retrieved "irrelevant text."

The deeper problem was:

> **The ranking system was allowing broad semantic similarity and section preferences to overpower the actual evidence requirements of the question.**

For a management-outlook question, a generic forward-looking disclaimer is not equivalent to management guidance or earnings-call commentary.

This exposed why retrieval diagnostics are essential.

---

# Another observed failure: numeric grounding

A Meta revenue test exposed a different issue.

The system retrieved useful-looking financial evidence, including:

- financial tables
- computed metrics
- earnings material
- structured income-statement data
- segment information

However, the generated answer reported:

> Revenue: **$39.07 billion**

while the verification layer flagged an unverified figure:

```text
WARNING unverified figures: {'$39'}
```

The source list and generated answer also showed inconsistencies between the requested year and some retrieved periods.

This demonstrates an important limitation:

> **Good retrieval does not automatically guarantee a correct financial answer.**

Numeric answers require stronger grounding and verification than ordinary narrative answers.

---

# Known limitations

Finsight is an engineering/research project, not a certified financial-data terminal.

## 1. Retrieval can still be wrong

Semantic similarity does not guarantee evidentiary relevance.

A document can contain the right vocabulary while answering a different question.

---

## 2. Ranking weights are not universally optimal

The retrieval pipeline contains task-aware weights and boosts.

These improve behavior for specific classes of questions, but overly aggressive boosts can create new ranking errors.

A ranking rule that helps one task may hurt another.

---

## 3. Document metadata can be imperfect

A section label such as `md&a`, `general`, or `earnings` is useful, but it is not proof that the chunk contains the required evidence.

Metadata should guide ranking rather than replace content-level relevance.

---

## 4. Financial numbers require verification

A language model can combine information from different periods or sources and produce a plausible-looking number.

That is particularly dangerous in finance because a plausible number can still be wrong.

Finsight therefore treats numeric verification as an important ongoing engineering problem.

---

## 5. Small local LLMs have limitations

The current Qwen 2.5 3B Instruct model is lightweight and practical for local experimentation, but smaller models can struggle with:

- long-context synthesis
- conflicting evidence
- numerical reasoning
- strict source attribution
- complex financial interpretation

The retrieval layer cannot completely compensate for model limitations.

---

## 6. Source coverage affects answer quality

If the correct evidence was never ingested, retrieval cannot return it.

The quality of the final answer therefore depends on:

```text
Data coverage
×
Chunk quality
×
Retrieval quality
×
Reranking quality
×
LLM synthesis
```

---

# What makes the project different

The interesting part of Finsight is not simply:

> "I connected an LLM to financial documents."

The project focuses on the harder engineering problem:

> **How do you build a financial RAG system that can tell you when its retrieval is wrong?**

The pipeline explicitly exposes intermediate decisions.

Instead of hiding everything behind:

```text
Question → LLM → Answer
```

Finsight exposes:

```text
Question
   ↓
Intent
   ↓
Candidate retrieval
   ↓
Hybrid ranking
   ↓
Section/source constraints
   ↓
Diversity
   ↓
Reranking
   ↓
Evidence selection
   ↓
LLM
   ↓
Verification
   ↓
Answer
```

That makes the system measurable, debuggable, and extensible.

---

# Project philosophy

### Evidence over confidence

A confident answer is not useful if the evidence is wrong.

### Retrieval over memorization

The LLM should work from current retrieved evidence rather than relying entirely on model memory.

### Specialized retrieval over one-size-fits-all search

Financial metrics, earnings commentary, risk analysis, trends, and news require different evidence.

### Observability over black-box behavior

Every stage should be inspectable.

### Verification over plausibility

Especially for financial figures, "sounds right" is not enough.

---

# Roadmap

The project is designed to evolve toward a more production-grade financial intelligence platform.

Planned areas include:

- retrieval evaluation datasets
- Recall@K / Precision@K / MRR / NDCG evaluation
- answer faithfulness evaluation
- stronger numerical verification
- source-to-claim mapping
- citation validation
- query classification benchmarking
- retrieval regression tests
- contradiction detection
- temporal consistency checks
- improved quarterly-period retrieval
- better evidence deduplication
- API serving with FastAPI
- Dockerized deployment
- AWS deployment
- latency and throughput monitoring
- evaluation dashboards

The long-term objective is not simply to make the model "sound smarter."

It is to make the **entire financial research pipeline more reliable and measurable**.

---

# Disclaimer

Finsight is a technical research and engineering project.

It is **not financial advice**, and generated outputs should not be treated as a substitute for professional investment research, primary-source verification, or qualified financial advice.

Financial figures, interpretations, and conclusions should be independently verified against the underlying filings and source material before being used for investment or business decisions.

---

# Author / Project

**Finsight — Financial Intelligence & Research Agent**

Built as an applied AI/ML engineering project focused on:

**RAG · Information Retrieval · Financial NLP · Hybrid Search · Reranking · LLMs · Evidence Grounding · Evaluation · Observability**

