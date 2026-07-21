# Configuration

Set environment variables before starting ingestion or the application.
Command-line launcher options apply to that application process.

| Variable | Default | Purpose |
| --- | --- | --- |
| `PMBOK_CHROMA_DIR` | `chroma_db` | Persistent vector database |
| `PMBOK_IMAGES_DIR` | `images` | Extracted and rendered source images |
| `PMBOK_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `PMBOK_EMBEDDING_REVISION` | Model default | Embedding revision |
| `PMBOK_RERANKER` | `cross-encoder/ms-marco-MiniLM-L-4-v2` | Cross-encoder model |
| `PMBOK_RERANKER_REVISION` | Model default | Cross-encoder revision |
| `PMBOK_LLM_MODEL` | `qwen2.5:3b` | Installed Ollama generation model |
| `PMBOK_OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `PMBOK_OLLAMA_TIMEOUT` | `600` | Generation request timeout in seconds |
| `PMBOK_KEEP_ALIVE` | `30m` | Ollama model retention duration |
| `PMBOK_LLM_CONTEXT` | `8192` | Model context window |
| `PMBOK_LLM_MAX_TOKENS` | `512` | Generated output cap |
| `PMBOK_CONTEXT_TOKENS` | `3000` | Evidence budget, measured conservatively in UTF-8 bytes |
| `PMBOK_CANDIDATE_K` | `10` | Candidates from each retrieval method |
| `PMBOK_TOP_K` | `3` | Reranked passages considered for context |
| `PMBOK_MIN_RERANK_SCORE` | `0.0` | Raw relevance-score threshold |
| `PMBOK_CHUNK_SIZE` | `800` | Maximum chunk characters |
| `PMBOK_CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `PMBOK_TESSERACT_CMD` | System `PATH` | Tesseract executable |
| `PMBOK_OCR_LANGUAGE` | `eng` | Installed OCR language |
| `PMBOK_OFFLINE` | `0` | Use cached Hugging Face models when set to `1` |

Changing the embedding model, its revision, or the indexed chunk configuration
requires a compatible index. Keep generation settings independent of ingestion
settings when comparing model configurations.

For offline use, run `src/download_models.py` before setting `PMBOK_OFFLINE=1`.
Ollama models are installed separately with `ollama pull`.

## Index snapshots

`active_index.json` identifies the current collection. Previous collections remain
available to existing readers. To perform index maintenance, stop readers and
back up the database directory before removing obsolete collections. Keep the
collection referenced by the active manifest.

`ingest.lock` prevents concurrent writers. After an interrupted process, confirm
the writer has stopped before removing a remaining lock file.
