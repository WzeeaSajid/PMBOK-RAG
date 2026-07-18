"""Comparison evidence coverage and compact structured responses."""
import json
import unittest

import _bootstrap

from retrieval import Reranker, comparison_terms, definition_evidence
from service import RAGService, select_context, validate_answer

from test_pipeline import chunk

DEFINITIONS = ('Chapter 5 − Project Scope Management\n\n'
               '•\nProduct scope. The features and functions that characterize a product,\nservice, or result\n\n'
               '•\nProject scope. The work that needs to be accomplished to deliver a product,\nservice, or result with the specified features and functions.')
COMPLETION = ('Completion of the project scope is measured against the project management plan, '
              'the project scope statement, and its associated WBS and WBS dictionary, '
              'but completion of the product scope is measured against the product requirements.')
QUESTION = 'How does project scope differ from product scope?'


class ComparisonTests(unittest.TestCase):
    def test_comparison_intents_are_generic(self):
        for query in (QUESTION, 'What is the difference between project scope and product scope?',
                      'Compare project scope with product scope'):
            self.assertEqual(comparison_terms(query), ('project scope', 'product scope'))
        self.assertEqual(comparison_terms('Compare risk avoidance and risk transfer'),
                         ('risk avoidance', 'risk transfer'))
        self.assertEqual(comparison_terms('What is project scope?'), ())

    def test_both_definition_paragraphs_are_available(self):
        spans = definition_evidence(QUESTION, chunk(DEFINITIONS))
        self.assertEqual({term for term, text in spans}, {'project scope', 'product scope'})
        self.assertTrue(all(text in ' '.join(DEFINITIONS.split()) for term, text in spans))

    def test_both_concepts_precede_completion_only_passage(self):
        class Model:
            def predict(self, pairs):
                return [7.09, 6.65]
        ranked = Reranker(Model()).rerank(
            QUESTION, [chunk(COMPLETION, page=119), chunk(DEFINITIONS, page=119)])
        self.assertEqual(ranked[0]['comparison_coverage'], 2)
        self.assertEqual(ranked[0]['text'], DEFINITIONS)
        self.assertEqual(ranked[0]['rerank_score'], 6.65)

    def test_complete_definition_excerpts_fit_cpu_budget(self):
        selected, records = select_context(
            [chunk(DEFINITIONS)], token_budget=1200, question=QUESTION)
        self.assertIn('Product scope.', records[0]['passage'])
        self.assertIn('Project scope.', records[0]['passage'])
        self.assertEqual(selected[0]['full_text'], DEFINITIONS)

    def test_flat_schema_answer_validates_quotes_and_ids(self):
        selected, _ = select_context([chunk(DEFINITIONS)], question=QUESTION)
        raw = json.dumps(dict(answer='Product scope describes features; project scope describes the work required to deliver them.',
                              evidence=[dict(source_id='S1', quote='The features and functions'),
                                        dict(source_id='S1', quote='The work that needs to be accomplished')]))
        statements, reason = validate_answer(raw, selected)
        self.assertEqual(reason, 'validated')
        self.assertEqual(statements[0]['source_ids'], ['S1'])

    def test_flat_schema_refusal_and_invalid_evidence(self):
        selected, _ = select_context([chunk(DEFINITIONS)])
        self.assertEqual(validate_answer(
            '{"answer":"","evidence":[]}', selected)[1], 'model_abstained')
        for evidence in ([dict(source_id='S99', quote='The features and functions')],
                         [dict(source_id='S1', quote='An invented quote that is absent')], [None]):
            raw = json.dumps(dict(answer='Comparison.', evidence=evidence))
            self.assertIsNone(validate_answer(raw, selected)[0])

    def test_quote_cannot_bridge_removed_source_text(self):
        selected = [dict(source_id='S1', text='First supporting phrase. Second supporting phrase.',
                         full_text='First supporting phrase. Omitted sentence. Second supporting phrase.')]
        raw = json.dumps(dict(answer='Comparison.', evidence=[
                         dict(source_id='S1', quote='phrase. Second supporting')]))
        self.assertEqual(validate_answer(raw, selected)[1], 'quote_not_in_source')

    def test_comparison_uses_generation_with_selected_evidence(self):
        class Retriever:
            def search(self, *args, **kwargs):
                return [chunk(DEFINITIONS, page=119)]

        class Ranker:
            def rerank(self, query, chunks, **kwargs):
                return [dict(c, rerank_score=6.65) for c in chunks]

        class LLM:
            called = False

            def generate(self, messages):
                self.called = True
                self.prompt = messages[-1]['content']
                return json.dumps(dict(answer='Product scope describes features; project scope describes required work.',
                                       evidence=[dict(source_id='S1', quote='The features and functions'),
                                                 dict(source_id='S1', quote='The work that needs to be accomplished')]))
        llm = LLM()
        result = RAGService(None, Retriever(), Ranker(), llm).answer(QUESTION)
        self.assertTrue(llm.called)
        self.assertFalse(result.refused)
        self.assertIn('Product scope.', llm.prompt)
        self.assertIn('Project scope.', llm.prompt)
        self.assertEqual(result.sources[0]['metadata']['page_number'], 119)


if __name__ == '__main__':
    unittest.main()
