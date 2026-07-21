# PMBOK RAG

Local, source-grounded question answering for PMBOK reference documents.
The application combines PDF extraction, hybrid retrieval, cross-encoder
reranking, and Ollama generation in a Chainlit interface.

## Features

- Native PDF text, table extraction, and Tesseract OCR for images and scanned pages.
- Page-aware chunks with document, edition, and printed-page metadata.
- Vector and BM25 retrieval with document filtering and reciprocal-rank fusion.
- Definition and comparison evidence selection using indexed source passages.
- Direct source extraction for complete definitions and generation for other questions.
- Structured answers with validated source IDs and supporting quotations.
- Shared models warmed at startup and reused across chat sessions.
- Atomic index publication when adding or replacing a document.

## Requirements

- Python 3.11–3.13
- Ollama with an installed text-generation model
- Tesseract OCR for scanned PDFs and image text

Run the commands below from the project directory. The examples use Windows
PowerShell. On Linux or macOS, create the environment with `python3 -m venv .venv`
and use `.venv/bin/python` in place of `.venv\Scripts\python.exe`.

## Installation

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src/download_models.py
ollama pull qwen2.5:3b
```

Start the Ollama application, or run `ollama serve` if the service is not running.
Tesseract is discovered through `PATH`. An explicit executable path can also be set:

```powershell
$env:PMBOK_TESSERACT_CMD = "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

## Index a document

Place the PDF at the path supplied to the ingestion command:

```powershell
.\.venv\Scripts\python.exe src/ingest.py data/PMBOK_Guide_3rd_Edition.pdf --doc-name pmbok3 --title "PMBOK Guide" --edition "3rd edition"
```

Use the same `--doc-name` when replacing a document. Other indexed documents are
retained. Reload the application after indexing to use the new snapshot.

Source PDFs, extracted images, model files, and local databases remain outside
version control.

## Run the application

```powershell
.\.venv\Scripts\python.exe scripts/run_app.py
```

Open [localhost:8000](http://localhost:8000). Each answer includes document and
page citations; expand a source to inspect its original text.

Model settings can be supplied per launch:

```powershell
.\.venv\Scripts\python.exe scripts/run_app.py --model qwen2.5:1.5b-instruct --context 4096 --evidence-budget 1200 --max-output 384
```

Install the selected model with `ollama pull` before launching it. Application
settings can also be supplied through environment variables; see
[configuration](docs/configuration.md).

## Command-line queries

```powershell
.\.venv\Scripts\python.exe src/query.py "What is a work breakdown structure?" --source-doc pmbok3
```

Omit the question to open the interactive command-line session.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q src scripts examples
```

The suite covers extraction, OCR, page attribution, document replacement,
retrieval, definition parsing, comparison evidence, citation validation, and
CLI/UI contracts. Integration tests use local dependencies when installed.
The included GitHub Actions workflow runs checks on Python 3.11, 3.12, and 3.13.

## Evaluation

Generate and index the included example document:

```powershell
.\.venv\Scripts\python.exe examples/create_sample_pdf.py
.\.venv\Scripts\python.exe src/ingest.py examples/sample.pdf --doc-name sample --title "Example document"
.\.venv\Scripts\python.exe src/evaluate.py examples/evaluation.jsonl --output evaluation_results.json
```

Use `--retrieval-only` to evaluate retrieval without generation. Evaluation reports
candidate recall, context recall, citation-page precision, answer-term recall,
refusal accuracy, and response-time statistics. See [evaluation](docs/evaluation.md)
for the dataset format.

## Project structure

| Path | Responsibility |
| --- | --- |
| `src/ingest.py` | PDF extraction, OCR, and chunk preparation |
| `src/storage.py` | Index compatibility and atomic snapshot publication |
| `src/retrieval.py` | Hybrid retrieval, reranking, and source-entry parsing |
| `src/service.py` | Evidence selection, answer generation, and validation |
| `src/chainlit_app.py` | Web interface lifecycle and source display |
| `src/query.py` | Command-line interface |
| `src/evaluate.py` | Labelled query evaluation |
| `scripts/run_app.py` | Configurable application launcher |
| `tests/` | Regression and integration tests |

See [architecture](docs/architecture.md) for the request and indexing flows.
