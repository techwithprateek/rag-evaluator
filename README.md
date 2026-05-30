# RAG Evaluator

> An end-to-end RAG pipeline with automated evaluation using the [RAGAS](https://docs.ragas.io/) framework — so you stop vibing and start measuring.

---

## Why You Can't Miss RAG Evaluation

Most people building RAG applications fall into the **vibe check trap** — they ask the chatbot a few questions, the answers look reasonable, and they ship it. But "looks reasonable" is not a metric.

Your RAG pipeline can fail silently in multiple ways:
- The LLM hallucinates facts not present in any retrieved chunk
- The answer is grounded in context but doesn't actually answer what was asked
- The retriever pulls irrelevant chunks, wasting context window and confusing the model
- The retriever misses key information, leaving the model unable to fully answer

You won't catch any of these with a vibe check. You need a systematic evaluation framework — and that's exactly what RAGAS provides.

---

## The 4 Metrics That Matter

### 🔒 Faithfulness
Checks whether **every claim in the generated answer can be traced back to the retrieved context**. A high faithfulness score means the model isn't hallucinating — it's only saying things supported by what was retrieved.

> *"Did the model make up anything?"*

### 🎯 Answer Relevancy
Answers can be perfectly faithful to context but still **not answer what was actually asked**. Answer Relevancy measures whether the response genuinely addresses the user's question.

> *"Did the model actually answer the question?"*

### 📐 Context Precision
Measures **what fraction of the retrieved chunks were actually needed** to answer the question. A low score means your retriever is pulling in noisy, irrelevant chunks — polluting the context window.

> *"Was the retrieved context precise or full of noise?"*

### 🔍 Context Recall
Checks whether **the retriever found all the information needed** to fully answer the question. A low score means your retriever is missing important chunks — the model doesn't have what it needs.

> *"Did the retriever find everything it needed to?"*

---

## RAGAS Framework Explained

RAGAS is an **LLM-as-a-judge** evaluation framework. Here's what makes it powerful:

- **No labels required at inference time** — it only looks at inputs and outputs, not how the pipeline was built
- **Model-agnostic** — works with any RAG architecture (LangChain, LlamaIndex, custom, etc.)
- **Uses an LLM as the evaluator** — this project uses the currently selected `PROVIDER` model (OpenAI `gpt-4o-mini` or HuggingFace `Mistral-7B-Instruct-v0.3`) to score each metric by reasoning over the question, answer, context, and ground truth
- **Reproducible** — the same dataset always gives you comparable scores, so you can track improvements over time

---

## Architecture

```
Wikipedia Articles (5 AI topics)
          │
          ▼
   [ ingest.py ]
   Chunk → Embed → ChromaDB
          │
          ▼
   [ rag_pipeline.py ]
   For each question in questions.json:
     Retrieve top-3 chunks → Generate answer → Save to rag_outputs.json
          │
          ▼
   [ evaluate.py ]
   Load rag_outputs.json → RAGAS scores each sample
     → results/ragas_results.csv
```

### RAG Pipeline
- **Documents**: 5 Wikipedia articles on AI topics (~952 chunks)
- **Chunking**: `RecursiveCharacterTextSplitter` — 500 characters, 50 overlap
- **Embeddings**: `text-embedding-3-small` (OpenAI) or `all-MiniLM-L6-v2` (HuggingFace)
- **Vector Store**: ChromaDB (local, no infra needed)
- **Retrieval**: Top-3 most similar chunks per question
- **LLM**: `gpt-4o-mini` (OpenAI) or `Mistral-7B-Instruct-v0.3` (HuggingFace)

### RAGAS Evaluation
- **Evaluator LLM**: the provider-selected LLM (`gpt-4o-mini` for OpenAI, `Mistral-7B-Instruct-v0.3` for HuggingFace) acting as judge
- **Input**: question + generated answer + retrieved contexts + ground truth
- **Output**: per-sample scores for all 4 metrics → averaged and saved to CSV

---

## Tech Stack

| Component | Technology |
|---|---|
| Framework | LangChain |
| Vector Store | ChromaDB (local) |
| Document Source | Wikipedia API |
| LLM (RAG) | OpenAI `gpt-4o-mini` / HuggingFace `Mistral-7B` |
| Embeddings | OpenAI `text-embedding-3-small` / `all-MiniLM-L6-v2` |
| Evaluation | RAGAS 0.4 |
| Evaluator LLM | Provider-selected: OpenAI `gpt-4o-mini` / HuggingFace `Mistral-7B-Instruct-v0.3` |
| Output Format | JSON (RAG outputs) + CSV (scores) |

---

## Project Structure

```
rag-evaluator/
├── data/questions.json        # 10 pre-generated Q&A pairs with ground truths
├── outputs/rag_outputs.json   # RAG pipeline outputs (auto-created)
├── results/ragas_results.csv  # RAGAS evaluation scores (auto-created)
├── chroma_db/                 # Local ChromaDB vector store (auto-created)
├── config.py                  # Provider config — switch OpenAI ↔ HuggingFace
├── ingest.py                  # Load Wikipedia → chunk → embed → ChromaDB
├── rag_pipeline.py            # Run RAG on all questions → save outputs
├── evaluate.py                # Run RAGAS evaluation → save results
├── requirements.txt
└── .env.example
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Configure environment**
```bash
cp .env.example .env
# Set PROVIDER and your API key
```

**.env example:**
```env
PROVIDER=openai
OPENAI_API_KEY=your_key_here
```

---

## Usage

Run the three scripts in order:

```bash
# Step 1: Load Wikipedia articles, chunk, embed, store in ChromaDB
python3 ingest.py

# Step 2: Run RAG pipeline on all questions, save outputs to JSON
python3 rag_pipeline.py

# Step 3: Evaluate with RAGAS, print summary, save scores to CSV
python3 evaluate.py
```

---

## Sample Results

```
==================================================
        RAGAS Evaluation Summary
==================================================
  Faithfulness:       0.9857
  Answer Relevancy:   0.9148
  Context Precision:  0.8917
  Context Recall:     0.7167
==================================================
```

---

## Topics Covered

Wikipedia articles used as the knowledge base:
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Natural Language Processing
- Large Language Model

---

## Provider Options

| Setting | LLM | Embeddings |
|---|---|---|
| `PROVIDER=openai` | `gpt-4o-mini` | `text-embedding-3-small` |
| `PROVIDER=huggingface` | `Mistral-7B-Instruct-v0.3` (HF Inference API) | `all-MiniLM-L6-v2` (local) |