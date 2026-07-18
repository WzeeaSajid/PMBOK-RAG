"""Regression coverage for the timeout shown in the Windows logs."""
import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import URLError

import _bootstrap

import service


class TimeoutTests(unittest.TestCase):
    def test_read_timeout_is_not_reported_as_server_stopped(self):
        def transport(*args, **kwargs):
            raise TimeoutError('timed out')
        with self.assertRaisesRegex(service.PipelineError, 'within 300 seconds') as caught:
            service.OllamaClient(transport).request('/api/chat', {}, timeout=300)
        self.assertNotIn('Start ollama serve', str(caught.exception))

    def test_wrapped_timeout_is_classified(self):
        def transport(*args, **kwargs):
            raise URLError(TimeoutError('timed out'))
        with self.assertRaisesRegex(service.PipelineError, 'timed out'):
            service.OllamaClient(transport).request('/api/chat')

    def test_connection_refused_retains_actual_cause(self):
        def transport(*args, **kwargs):
            raise URLError(ConnectionRefusedError('connection refused'))
        with self.assertRaisesRegex(service.PipelineError, 'connection refused'):
            service.OllamaClient(transport).request('/api/chat')

    def test_timeout_and_output_are_configurable(self):
        seen = {}

        def transport(request, timeout):
            seen.update(timeout=timeout, payload=json.loads(request.data))
            return io.BytesIO(b'{"message":{"content":"{}"},"done_reason":"stop"}')
        with patch.dict(os.environ, {'PMBOK_OLLAMA_TIMEOUT': '700', 'PMBOK_LLM_MAX_TOKENS': '256', 'PMBOK_KEEP_ALIVE': '15m'}):
            service.OllamaClient(transport).generate([])
        self.assertEqual(seen['timeout'], 700)
        self.assertEqual(seen['payload']['options']['num_predict'], 256)
        self.assertEqual(seen['payload']['keep_alive'], '15m')

    def test_generation_truncation_has_actionable_error(self):
        def transport(*args, **kwargs):
            return io.BytesIO(b'{"message":{"content":"{"},"done_reason":"length"}')
        with self.assertRaisesRegex(service.PipelineError, 'token limit'):
            service.OllamaClient(transport).generate([])


if __name__ == '__main__':
    unittest.main()
