# Evaluation

Evaluation cases are stored as JSON Lines. Each record identifies the question,
expected document/page evidence, answer terms, and answerability label.

```json
{"question":"How is the cost performance index calculated?","source_doc":"sample","unanswerable":false,"expected_sources":[{"source_doc":"sample","page_number":2}],"expected_answer_terms":["earned value","actual cost"]}
```

Page labels use one-based PDF page numbers. `source_doc` matches the name supplied
to ingestion. Use `unanswerable: true` for questions intentionally outside the
indexed document and an empty `expected_sources` array for those cases.

```powershell
.\.venv\Scripts\python.exe src/evaluate.py examples/evaluation.jsonl --output evaluation_results.json
.\.venv\Scripts\python.exe src/evaluate.py examples/evaluation.jsonl --retrieval-only --output evaluation_results_retrieval.json
```

| Metric | Calculation |
| --- | --- |
| Candidate recall | Expected source pages present in the candidate pool |
| Context recall | Expected source pages included in selected evidence |
| Citation-page precision | Cited pages matching the labelled source pages |
| Answer-term recall | Labelled terms present in the answer |
| Refusal accuracy | Predicted refusal matching the answerability label |
| Median response time | Median measured request duration |
| P95 response time | 95th-percentile measured request duration |

Each case includes its result and error status. The summary reports errors
separately. Use `--min-score` to compare reranking thresholds against the same
labelled cases. Retain the document edition, model configuration, and dataset
when comparing runs.
