# Agentic RAG Chatbot — "Attention Is All You Need"

An Agentic RAG (Retrieval-Augmented Generation) pipeline built with LangGraph 
that can autonomously retrieve, evaluate, and synthesize information from 
the original Transformer paper.

## Agentic patterns

- **Adaptive RAG** — router decides retrieval vs. direct answer
- **Corrective RAG** — document grading + query rewriting with retry loops
- **Self-RAG** — hallucination check + answer grading

## Quick start

Open in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/benyo1203/agentic-rag-arxiv/blob/main/notebooks/agentic_rag.ipynb)

Run locally:

```bash
git clone https://github.com/benyo1203/agentic-rag-arxiv.git
cd agentic-rag-arxiv
pip install -r requirements.txt
jupyter notebook notebooks/agentic_rag.ipynb
```

## Tech stack

- **LangGraph** — state machine for the agentic pipeline
- **ChromaDB** — vector store with metadata filtering
- **sentence-transformers** (all-MiniLM-L6-v2) — 384-dim embeddings
- **flan-t5-large** — generation and classification (CPU-friendly)
- **PyMuPDF** — PDF text extraction

## Limitations

The current prototype uses flan-t5-large due to compute constraints. 
The pipeline is model-agnostic — swapping to Mistral-7B or Llama-3-8B 
with GPU access would significantly improve answer quality. 
See the notebook's Section 8 for detailed bottleneck analysis.
