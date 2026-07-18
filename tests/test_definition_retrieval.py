"""Regression for publication-history passages displacing term definitions."""
import unittest

import _bootstrap

from retrieval import HybridRetriever, Reranker, focused_query, is_direct_definition, clean_for_search

from test_pipeline import Collection, Embedder, FakeBM25, chunk

HISTORY = ('B.4\n2000 Update\nThis document superseded the PMBOK Guide. '
           'The scope of the project using the 1996 publication as its starting point was to add new material.')
DEFINITION = 'Project Scope. The work required to deliver a product with the specified features and functions.'
CAPTION = ('Figure 5-1. Project Scope Management Overview\n'
           'A Guide to the Project Management Body of Knowledge (PMBOK\n'
           '® Guide) Third Edition\n'
           '2004 Project Management Institute, Four Campus Boulevard, Newtown Square, PA 19073-3299 USA\n'
           '105\nNAVIGATION')


class DefinitionTests(unittest.TestCase):
    def test_attribution_is_removed_without_changing_target_concept(self):
        self.assertEqual(focused_query(
            'What is project scope according to the PMBOK Guide?'), 'What is project scope?')
        self.assertEqual(focused_query(
            'According to the PMBOK Guide, what is project scope?'), 'what is project scope?')
        self.assertEqual(focused_query('How did the 2000 update change the guide?'),
                         'How did the 2000 update change the guide?')

    def test_whole_term_definition_not_scope_statement_or_history(self):
        question = 'What is project scope according to the PMBOK Guide?'
        self.assertTrue(is_direct_definition(question, chunk(DEFINITION)))
        self.assertFalse(is_direct_definition(question, chunk(HISTORY)))
        self.assertFalse(is_direct_definition(question, chunk(CAPTION)))
        self.assertFalse(is_direct_definition(question, chunk(
            'Project Scope Statement. A narrative description.')))
        self.assertFalse(is_direct_definition(
            'How was the PMBOK Guide updated?', chunk(DEFINITION)))

    def test_multiline_and_glossary_metadata_definitions(self):
        self.assertTrue(is_direct_definition('Define project scope', chunk(
            'Project Scope\nThe work required for delivery.')))
        self.assertTrue(is_direct_definition('Define project scope', chunk(
            'Project Scope.\nThe work required for delivery.')))
        item = chunk('Work required for delivery.', kind='glossary')
        item['metadata']['term'] = 'Project Scope'
        self.assertTrue(is_direct_definition('What is project scope?', item))

    def test_footer_is_removed_for_ranking_but_original_remains_intact(self):
        c = chunk(CAPTION)
        cleaned = clean_for_search(c['text'])
        self.assertNotIn('Four Campus', cleaned)
        self.assertNotIn('Body of Knowledge', cleaned)
        self.assertIn('Project Scope Management Overview', cleaned)
        self.assertEqual(c['text'], CAPTION)

    def test_definition_supplements_candidate_pool_and_respects_document_filter(self):
        collection = Collection()
        chunks = [chunk(HISTORY + (' project scope' * 10), source='a', page=331),
                  chunk(DEFINITION, source='a', page=385), chunk(DEFINITION, source='b', page=1)]
        collection.add(ids=[c['id'] for c in chunks], documents=[c['text'] for c in chunks],
                       metadatas=[c['metadata'] for c in chunks], embeddings=[[0, 1]] * len(chunks))
        hits = HybridRetriever(collection, FakeBM25).search('What is project scope according to the PMBOK Guide?',
                                                            Embedder(), top_k=1, source_doc='a')
        self.assertTrue(any(c['metadata']['page_number'] == 385 for c in hits))
        self.assertTrue(all(c['metadata']['source_doc'] == 'a' for c in hits))

    def test_definition_precedes_high_scoring_history_for_definition_question(self):
        class Model:
            def predict(self, pairs):
                self.pairs = pairs
                return [6.63, 6.24, 4.8]
        model = Model()
        ranked = Reranker(model).rerank('What is project scope according to the PMBOK Guide?',
                                        [chunk(HISTORY, page=331), chunk(CAPTION, page=120), chunk(DEFINITION, page=385)])
        self.assertEqual(ranked[0]['metadata']['page_number'], 385)
        self.assertEqual(ranked[0]['rerank_score'], 4.8)  # No invented confidence score.
        self.assertEqual(model.pairs[0][0], 'What is project scope?')
        self.assertNotIn('Body of Knowledge', model.pairs[1][1])

    def test_unrelated_definition_does_not_bypass_relevance_gate(self):
        class Model:
            def predict(self, pairs):
                return [4, -20]
        ranked = Reranker(Model()).rerank('What is project scope?',
                                          [chunk(HISTORY), chunk(DEFINITION)])
        self.assertFalse(ranked[0]['definition_match'])

    def test_definition_is_the_first_complete_source_with_cpu_budget(self):
        from service import select_context

        class Model:
            def predict(self, pairs):
                return [6.63, 4.8]
        ranked = Reranker(Model()).rerank('What is project scope according to the PMBOK Guide?',
                                          [chunk(HISTORY, page=331), chunk(DEFINITION, page=385)])
        selected, records = select_context(ranked, token_budget=1200)
        self.assertEqual(selected[0]['metadata']['page_number'], 385)
        self.assertEqual(records[0]['passage'], DEFINITION)


if __name__ == '__main__':
    unittest.main()
