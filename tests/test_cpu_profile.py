"""Startup reuse and compact prompt regressions."""
import io
import json
import os
import unittest
from unittest.mock import MagicMock, patch

import _bootstrap

import config
import service


class CPUProfileTests(unittest.TestCase):
    def test_warmup_uses_same_context_and_keepalive_as_generation(self):
        requests = []

        def transport(request, timeout):
            if request.full_url.endswith('/api/tags'):
                return io.BytesIO(json.dumps({'models': [{'name': config.LLM_MODEL}]}).encode())
            requests.append(json.loads(request.data))
            return io.BytesIO(b'{"message":{"content":"{}"}}')
        with patch.dict(os.environ, {'PMBOK_KEEP_ALIVE': '-1'}), patch.object(config, 'LLM_CONTEXT_TOKENS', 4096):
            client = service.OllamaClient(transport)
            client.preflight()
            client.generate([])
        self.assertEqual(len(requests), 2)
        self.assertTrue(all(r['options']['num_ctx'] == 4096 for r in requests))
        self.assertTrue(all(r['keep_alive'] == -1 for r in requests))
        self.assertEqual(requests[0]['options']['num_predict'], 1)
        self.assertEqual(requests[0]['messages'][0]['content'], service.SYSTEM_PROMPT)

    def test_bookkeeping_stays_out_of_model_prompt(self):
        chunk = dict(id='abc', text='Scope is the work required to deliver a product.', metadata=dict(
            source_doc='pmbok3', document_title='PMBOK', edition='3rd', page_number=385,
            file_sha256='x' * 64, image_path='C:/private/figure.png', char_start=200, char_end=280))
        selected, records = service.select_context([chunk])
        prompt = service.build_messages('What is scope?', records)[1]['content']
        self.assertNotIn('sha256', prompt)
        self.assertNotIn('private', prompt)
        self.assertNotIn('char_start', prompt)
        self.assertIn('Scope is the work required', prompt)
        self.assertEqual(selected[0]['metadata']['page_number'], 385)
        self.assertEqual(records[0]['document'], 'PMBOK')

    def test_startup_reuses_models_retriever_and_ollama_client(self):
        state = {'collection': 'snapshot1', 'signature': config.index_signature()}
        collection = MagicMock(name='collection')
        collection.name = 'snapshot1'
        service._service_for_snapshot.cache_clear()
        try:
            with patch.object(service, 'read_state', return_value=state), \
                    patch.object(service, 'load_collection', return_value=collection), \
                    patch.object(service, 'load_embedding_model') as embed, \
                    patch.object(service, 'load_reranker') as rerank, \
                    patch.object(service, 'HybridRetriever') as retrieve, \
                    patch.object(service, 'OllamaClient') as llm:
                first = service.get_service()
                second = service.get_service()
                self.assertIs(first, second)
                embed.assert_called_once()
                rerank.assert_called_once()
                retrieve.return_value.prepare.assert_called_once_with()
                llm.return_value.preflight.assert_called_once()
        finally:
            service._service_for_snapshot.cache_clear()


if __name__ == '__main__':
    unittest.main()
