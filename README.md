# Document Indexing & Retrieval

A Python pipeline that extracts text from PDF/DOCX files, splits it into
chunks using multiple strategies, generates embeddings with the Gemini API,
and stores everything in PostgreSQL with pgvector for semantic search.

## Prerequisites

- Python 3.11+
- Docker Desktop (for the local PostgreSQL + pgvector instance)
- A Gemini API key ([Google AI Studio](https://aistudio.google.com/apikey))

## Installation

```bash
git clone https://github.com/amitnahum18/Task_2_For_Jeen.git
cd Task_2_For_Jeen
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and fill in the values. Never commit `.env`.

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | API key for the Gemini API, used to generate embeddings. |
| `POSTGRES_URL` | PostgreSQL connection string, e.g. `postgresql://user:password@localhost:5432/document_indexer`. |
| `EMBEDDING_DIM` | Output dimensionality requested from `gemini-embedding-001` (default `768`). Must match the vector column size in the database schema. |

## Database

This project uses a managed PostgreSQL instance with the [pgvector](https://github.com/pgvector/pgvector) extension (e.g. [Supabase](https://supabase.com) or [Neon](https://neon.tech), both offer a free tier):

1. Create a project with either provider.
2. Enable the `vector` extension (Supabase: Database → Extensions → `vector`; Neon: run `CREATE EXTENSION IF NOT EXISTS vector;` in the SQL editor).
3. Copy the connection string into `POSTGRES_URL` in your `.env` file.
