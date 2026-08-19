# Document Indexing & Retrieval

A Python pipeline that extracts text from PDF/DOCX files, splits it into
chunks using multiple strategies, generates embeddings with the Gemini API,
and stores everything in PostgreSQL with pgvector for semantic search.

## Project Structure

```
index_documents.py          CLI: index a document
search.py                   CLI: semantic search
document_indexer/
  config.py                 .env loading and validation
  exceptions.py              custom exception hierarchy
  extraction.py              PDF/DOCX -> clean text
  chunking.py                 fixed_size / sentence / paragraph strategies
  embeddings.py               Gemini API wrapper
  db.py                       PostgreSQL + pgvector: schema, insert, search
docs/                        sample PDF/DOCX used in the run examples below
tests/
  unit/                       mocked, no network or database needed
  integration/                real-database rigor suite (pytest -m integration)
```

## Prerequisites

- Python 3.10+ (developed and tested on 3.12.3)
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

`document_indexer.db.init_schema()` creates the rest automatically (idempotent - safe to run on every startup): the `vector` extension, the `document_chunks` table, and an HNSW cosine-distance index.

| Column | Type |
|---|---|
| `id` | `SERIAL PRIMARY KEY` |
| `chunk_text` | `TEXT NOT NULL` |
| `embedding` | `VECTOR(EMBEDDING_DIM) NOT NULL` |
| `filename` | `TEXT NOT NULL` |
| `split_strategy` | `TEXT NOT NULL` |
| `created_at` | `TIMESTAMPTZ NOT NULL DEFAULT now()` |

## Usage

Index a document:

```bash
python index_documents.py --file ./docs/example.pdf --strategy paragraph
```

```
Extracted 2378 characters -> 5 chunks (paragraph)
Indexed 5 chunks from 'example.pdf' into the database.
```

Search it:

```bash
python search.py --query "login issue"
```

```
1. distance=0.3226  example.pdf (paragraph)
   Acme Cloud Storage - Support Guide
If you are unable to log into your account, first verify that you are using the correct email address and
password. Passwords are case sensitive and must be at least...
2. distance=0.3381  example.pdf (paragraph)
   Account access problems can occur when a user is removed from a shared workspace or when
two-factor authentication is misconfigured. Workspace administrators can restore access from the
Members page....
3. distance=0.3776  example.pdf (paragraph)
   Technical issues such as slow uploads or failed synchronization are usually caused by an unstable
internet connection or an outdated client application. Start by checking your connection speed and
res...
```

(Full run returns 5 results, one per indexed chunk, ranked by ascending cosine distance - the login-related paragraph correctly comes out on top for a "login issue" query. `search.py` also accepts `--top-k N`; `index_documents.py` also accepts `--chunk-size` and, for the `fixed_size` strategy, `--overlap`.)

If the table is empty or nothing is indexed yet, `search.py` prints `No results found.` and exits `0` rather than erroring.

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

## Embeddings

`document_indexer.embeddings` calls the Gemini API (`gemini-embedding-001`) via the official `google-genai` SDK:

- `embed_chunks(texts, api_key, dimensions)` - batches all chunks from a document into a single API call, using task type `RETRIEVAL_DOCUMENT`.
- `embed_query(text, api_key, dimensions)` - embeds a search query using task type `RETRIEVAL_QUERY`, which the Gemini model uses to optimize the vector for matching against documents rather than other queries.
- `dimensions` is requested via the API's `output_dimensionality` parameter and must match `EMBEDDING_DIM` / the database's vector column size. Every returned embedding is checked against the expected dimension before being handed back to the caller.

## Python API Reference

Every module is usable directly, not only through the two CLI scripts.

**`document_indexer.config`**
- `load_config() -> Config` - reads and validates `GEMINI_API_KEY`, `POSTGRES_URL`, `EMBEDDING_DIM` from the environment. Raises `ConfigError` if something required is missing or malformed. `Config`'s `repr()` masks the key and connection string so they never leak into logs or tracebacks.

**`document_indexer.extraction`**
- `extract_text(file_path) -> str` - see [Text Extraction](#text-extraction).

**`document_indexer.chunking`**
- `chunk_text(text, strategy, **kwargs) -> list[str]` - dispatches to one of the three functions below by strategy name; see [Chunking Strategies](#chunking-strategies).
- `chunk_fixed_size(text, chunk_size=1000, overlap=200) -> list[str]`
- `chunk_by_sentences(text, chunk_size=1000) -> list[str]`
- `chunk_by_paragraphs(text) -> list[str]`
- `split_sentences(text) -> list[str]` - the sentence-boundary detector behind `chunk_by_sentences`; also usable standalone.

**`document_indexer.embeddings`**
- `embed_chunks(texts, api_key, dimensions) -> list[list[float]]` - see [Embeddings](#embeddings).
- `embed_query(text, api_key, dimensions) -> list[float]`

**`document_indexer.db`**
- `connect(postgres_url)` - context manager yielding a psycopg2 connection with the pgvector adapter registered; wraps connection failures as `DatabaseConnectionError`.
- `init_schema(conn, dimensions, table_name="document_chunks")` - creates the `vector` extension, the table, and an HNSW cosine-distance index (idempotent).
- `insert_chunks(conn, records: list[ChunkRecord], table_name="document_chunks")` - batch insert; no-op on an empty list.
- `search(conn, query_embedding, top_k=5, table_name="document_chunks") -> list[SearchResult]` - see [Database](#database) and the rigor suite below.
- `ChunkRecord(chunk_text, embedding, filename, split_strategy)` - one chunk ready to store.
- `SearchResult(id, chunk_text, filename, split_strategy, created_at, distance)` - one search hit.

`table_name` always defaults to `document_chunks`; it exists only so the integration test suite can point each test at its own disposable table. The CLI scripts never override it.

## Error Handling

All custom exceptions below inherit from `document_indexer.exceptions.DocumentIndexerError`, which is what both CLI scripts catch at the top level to print a clean `Error: ...` message and exit `1` instead of showing a raw traceback.

| Condition | Behavior |
|---|---|
| Required `.env` variable missing or invalid (e.g. non-numeric `EMBEDDING_DIM`) | `ConfigError` |
| File does not exist | `FileNotFoundError` |
| Extension is not `.pdf`/`.docx` | `UnsupportedFileTypeError` |
| Document has no extractable text, or is corrupted/unreadable despite a valid extension (e.g. a scanned PDF with no text layer, an empty file, or a file that isn't actually a valid PDF/DOCX) | `NoExtractableTextError` |
| Invalid chunking parameters (e.g. `overlap >= chunk_size`, non-positive `chunk_size`) | `ValueError` |
| Unknown `--strategy` value | `ValueError` |
| Gemini API call fails (network, auth, quota) or returns an unexpected embedding size | `EmbeddingGenerationError` |
| Database connection cannot be established | `DatabaseConnectionError` |
| pgvector rejects a record or query (bad dimension, `NaN`/`Infinity`, `NULL` embedding) | `InvalidEmbeddingError` |
| Search on an empty table | Empty result list (not an error) |

## Testing

```bash
pytest                  # unit tests only (no external services needed)
pytest -m integration   # also run the real-database rigor suite below (needs POSTGRES_URL)
```

Unit tests generate their own minimal PDF/DOCX fixtures on the fly (via `tmp_path`) rather than depending on `docs/example.*`, mock the Gemini client, and mock psycopg2 - none of them touch a network or a real database, so they run in about a second.

### Invalid-vector rigor suite (`tests/integration/test_db_integration.py`)

Mocks can't reproduce pgvector's actual constraint enforcement, so this suite runs against a real database (a fresh, uniquely-named table per test, dropped afterward) and asserts on pgvector's real behavior rather than assumed behavior:

| Case | Verified behavior |
|---|---|
| `NULL` embedding | Rejected by the `NOT NULL` constraint -> `InvalidEmbeddingError` |
| Wrong-dimension embedding (e.g. 5 values into a `VECTOR(3)` column) | Rejected by pgvector's own type check -> `InvalidEmbeddingError` |
| `NaN` in an embedding | Rejected at insert time ("NaN not allowed in vector") -> `InvalidEmbeddingError` |
| `Infinity` in an embedding | Rejected at insert time -> `InvalidEmbeddingError` |
| One bad record in a batch insert | The whole batch fails together (single statement) rather than partially inserting |
| Search query with the wrong dimension | Rejected -> `InvalidEmbeddingError` |
| A stored zero vector (the one vector pgvector *will* store that's still problematic - cosine distance against it is mathematically undefined) | `search()` explicitly filters out non-finite (`NaN`) distances rather than crashing or sorting them arbitrarily |
| Search on an empty table | Returns `[]`, no error |

Because NULL/NaN/Infinity/wrong-dimension are all rejected at insert time, a "corrupted row" can only ever reach the table as a mathematically-degenerate-but-valid vector (like all-zeros) - not as literally malformed data. The suite reflects that rather than testing for states pgvector makes unreachable.
