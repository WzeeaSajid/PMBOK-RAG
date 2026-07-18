"""Evaluate labelled JSONL questions against your local index and models."""
import argparse
import json
import statistics
import time
from pathlib import Path


def evaluate_dataset(pipeline, rows, retrieval_only=False):
    results = []
    for row in rows:
        started = time.monotonic()
        expected = {(s['source_doc'], int(s['page_number']))
                    for s in row.get('expected_sources', [])}
        result = dict(question=row['question'],
                      expected_unanswerable=row.get('unanswerable', False))
        try:
            if retrieval_only:
                candidates, ranked, selected, records = pipeline.retrieve(
                    row['question'], row.get('source_doc'))
                answer = None
                ids = [c['id'] for c in selected]
            else:
                answer = pipeline.answer(row['question'], row.get('source_doc'))
                ids = answer.diagnostics.get('retrieved_ids', [])
                # Retrieve the snapshot's passage metadata for recall scoring.
                lookup_ids = ids + answer.diagnostics.get('candidate_ids', [])
                by_id = pipeline.retriever.metadata_for(lookup_ids, row.get('source_doc'))
                selected = [by_id[cid] for cid in ids if cid in by_id]
                candidates = [by_id[cid]
                              for cid in answer.diagnostics.get('candidate_ids', []) if cid in by_id]

            def pages(chunks):
                return {(c['metadata']['source_doc'], int(c['metadata']['page_number'])) for c in chunks}
            result['retrieved_ids'] = ids
            result['candidate_recall'] = len(expected & pages(
                candidates)) / len(expected) if expected else None
            result['context_recall'] = len(expected & pages(
                selected)) / len(expected) if expected else None
            if answer is not None:
                result.update(answer=answer.answer, refused=answer.refused,
                              refusal_correct=answer.refused == row.get('unanswerable', False),
                              reason=answer.diagnostics.get('reason'))
                cited = pages(answer.sources)
                result['citation_page_precision'] = len(
                    cited & expected) / len(cited) if cited and expected else None
                terms = row.get('expected_answer_terms', [])
                result['answer_term_recall'] = sum(
                    term.casefold() in answer.answer.casefold() for term in terms) / len(terms) if terms else None
            result['error'] = None
        except Exception as exc:
            result['error'] = str(exc)
        result['elapsed_seconds'] = time.monotonic() - started
        results.append(result)

    def mean(key):
        values = [r[key] for r in results if r.get(key) is not None]
        return statistics.mean(values) if values else None
    timings = sorted(r['elapsed_seconds'] for r in results)
    summary = dict(question_count=len(results), errors=sum(r['error'] is not None for r in results),
                   candidate_recall=mean('candidate_recall'), context_recall=mean('context_recall'),
                   citation_page_precision=mean('citation_page_precision'), refusal_accuracy=mean('refusal_correct'),
                   answer_term_recall=mean('answer_term_recall'),
                   median_seconds=statistics.median(timings) if timings else None,
                   p95_seconds=timings[min(len(timings) - 1, int(len(timings) * .95))] if timings else None)
    return dict(mode='retrieval_only' if retrieval_only else 'full', summary=summary, results=results,
                metric_definitions=dict(citation_page_precision='Fraction of cited pages matching the labelled source pages.',
                                        answer_term_recall='Fraction of labelled answer terms present in the answer.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--retrieval-only', action='store_true')
    parser.add_argument('--min-score', type=float,
                        help='Override reranker threshold for this evaluation run.')
    parser.add_argument('--output', type=Path, default=Path('evaluation_results.json'))
    args = parser.parse_args()
    import config
    from service import get_service
    if args.min_score is not None:
        config.MIN_RERANK_SCORE = args.min_score
    rows = [json.loads(line) for line in args.dataset.read_text(
        encoding='utf-8').splitlines() if line.strip()]
    report = evaluate_dataset(get_service(
        require_llm=not args.retrieval_only), rows, args.retrieval_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))
    if report['summary']['errors']:
        parser.exit(1, 'Some questions failed; inspect the report.\n')


if __name__ == '__main__':
    main()
