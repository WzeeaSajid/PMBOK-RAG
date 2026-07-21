# Architecture

## Indexing

PyMuPDF extracts text blocks, font metadata, embedded images, and table rows.
Scanned pages and vector artwork can be rendered for Tesseract OCR. Source images
are namespaced by document hash, and successful OCR results are cached.

Text is grouped by page and section before splitting. Chunks retain their source
document, edition, PDF page, printed-page label, and local character spans. Entries
from the glossary are stored alongside body text, tables, and OCR passages.

Ingestion prepares a new Chroma collection containing the updated document and
unchanged documents. An atomic manifest update publishes the completed snapshot.
Existing query sessions retain their original snapshot until reloaded. The
manifest records the embedding model and index configuration.

## Retrieval

Vector search and BM25 run within the requested document scope. Candidate IDs are
merged through reciprocal-rank fusion and scored by a cross-encoder. Search
normalization removes source-attribution wrappers and running-footer artifacts.

Definition questions supplement the candidate pool with matching source entries.
The same parser is used for ranking and extractive answers, including acronym
aliases and glossary annotations. Entry-boundary checks distinguish headings
from wrapped prose.

Comparison questions identify the requested concepts and prioritize passages
covering both. Literal definition paragraphs can be selected from longer chunks
so the context budget covers the requested concepts. Full source text remains
available for citation display and quote checks.

## Answers

Complete definitions are returned as literal source spans with normalized
whitespace and their original page citations. Other requests use Ollama to
produce an answer and short supporting quotations in a compact JSON response.

Validation checks response structure, source IDs, and quoted spans. Quotes must
occur in both the supplied evidence and the original source text. The interface
adds citation labels from stored metadata rather than relying on generated page
numbers.

The web interface and command-line interface use the same service. Models and
the lexical index are initialized at startup and cached within the application
process. Blocking model and network operations run outside the UI event loop.
