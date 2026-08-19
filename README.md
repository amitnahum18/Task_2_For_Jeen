# Document Indexing & Retrieval

A Python pipeline that extracts text from PDF/DOCX files, splits it into
chunks using multiple strategies, generates embeddings with the Gemini API,
and stores everything in PostgreSQL with pgvector for semantic search.

## Prerequisites

- Python 3.11+
- A PostgreSQL database with the pgvector extension available (see [Database](#database))
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

## Text Extraction

`document_indexer.extraction.extract_text()` accepts a `.pdf` or `.docx` path and returns cleaned, whitespace-normalized text, with paragraph breaks kept as blank lines wherever the source format exposes them:

- **DOCX**: each paragraph in the document maps to one paragraph in the output (via `python-docx`), so paragraph-based chunking (see below) is exact.
- **PDF**: text is extracted per page with `pypdf`. PDFs carry no reliable, generator-independent signal for where a paragraph ends versus where a line merely wrapped, so page boundaries - the one paragraph-like break PDFs expose reliably - are used instead. Paragraph-based chunking on a PDF is therefore page-grained rather than true paragraph-grained.

Sample fixtures are provided in `docs/` (`example.pdf`, `example.docx`) - a short multi-section support guide used in the run examples below.

## Chunking Strategies

`document_indexer.chunking.chunk_text(text, strategy, **kwargs)` supports three strategies, selected with `--strategy` on the CLI:

| Strategy | How it splits | Key parameters |
|---|---|---|
| `fixed_size` | Fixed-width character windows, each overlapping the previous one | `chunk_size` (default 1000), `overlap` (default 200) |
| `sentence` | Sentences (regex + capitalization heuristic, with guards for common abbreviations like "Dr." or "e.g.") are greedily packed into chunks up to a size limit without splitting mid-sentence | `chunk_size` (default 1000) |
| `paragraph` | One chunk per paragraph (blank-line separated); see [Text Extraction](#text-extraction) for how paragraph boundaries are determined per file type | - |

The sentence splitter is a lightweight heuristic, not a trained model - it can still mis-split on unusual abbreviations it doesn't recognize, which just yields a slightly-off chunk boundary rather than a failure.

## Error Handling

| Condition | Behavior |
|---|---|
| File does not exist | `FileNotFoundError` |
| Extension is not `.pdf`/`.docx` | `UnsupportedFileTypeError` |
| Document has no extractable text (e.g. a scanned PDF with no text layer) | `NoExtractableTextError` |
| Invalid chunking parameters (e.g. `overlap >= chunk_size`, non-positive `chunk_size`) | `ValueError` |
| Unknown `--strategy` value | `ValueError` |

## Testing

```bash
pytest
```

Unit tests generate their own minimal PDF/DOCX fixtures on the fly (via `tmp_path`) rather than depending on `docs/example.*`, so each test controls exactly the input it needs.
