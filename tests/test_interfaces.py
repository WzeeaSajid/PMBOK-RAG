"""CLI/UI/evaluation contract tests without network or model downloads."""
import asyncio
import importlib.util
import io
import sys
import threading
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import _bootstrap

import evaluate
import query
import service

from test_pipeline import TemporaryIndex, Embedder, chunk


class EvaluationTests(unittest.TestCase):
    def pipeline(self):
        c = chunk('Scope is the work required.', source='sample', page=1)

        class Pipeline:
            def retrieve(self, *args):
                return [c], [dict(c, rerank_score=5)], [dict(c, source_id='S1')], []
        return Pipeline()

    def test_retrieval_metrics_match_ground_truth(self):
        rows = [dict(question='Scope?', expected_sources=[dict(source_doc='sample', page_number=1)]),
                dict(question='Risk?', expected_sources=[dict(source_doc='sample', page_number=2)])]
        report = evaluate.evaluate_dataset(self.pipeline(), rows, retrieval_only=True)
        self.assertEqual(report['summary']['context_recall'], .5)
        self.assertEqual(report['summary']['candidate_recall'], .5)
        self.assertIsNone(report['summary']['refusal_accuracy'])

    def test_errors_count_separately(self):
        class Broken:
            def retrieve(self, *args):
                raise RuntimeError('failure')
        report = evaluate.evaluate_dataset(Broken(), [dict(question='Q')], retrieval_only=True)
        self.assertEqual(report['summary']['errors'], 1)
        self.assertIsNone(report['summary']['context_recall'])

    def test_full_metrics_and_abstention(self):
        passage = chunk('Scope is the work required.', source='sample', page=1)

        class Retriever:
            def metadata_for(self, ids, source_doc=None):
                return {passage['id']: passage}

        class Pipeline:
            retriever = Retriever()

            def answer(self, question, source_doc):
                if question == 'Unanswerable':
                    return service.Answer(service.REFUSAL, refused=True)
                return service.Answer('Scope is work [S1]', [passage], diagnostics=dict(
                    candidate_ids=[passage['id']], retrieved_ids=[passage['id']]))
        rows = [dict(question='Scope', expected_sources=[dict(source_doc='sample', page_number=1)],
                     expected_answer_terms=['work']), dict(question='Unanswerable', unanswerable=True)]
        result = evaluate.evaluate_dataset(Pipeline(), rows)
        self.assertEqual(result['summary']['refusal_accuracy'], 1)
        self.assertEqual(result['summary']['citation_page_precision'], 1)
        self.assertEqual(result['summary']['answer_term_recall'], 1)


class CLITests(unittest.TestCase):
    def test_cli_delegates_filter_and_formats_source(self):
        class Pipeline:
            called = False

            def answer(self, question, source_doc):
                self.called = question == 'Scope?' and source_doc == 'sample'
                return service.Answer('Required work [S1]', [dict(chunk(source='sample'), source_id='S1')])
        pipeline = Pipeline()
        with patch.object(query, 'get_service', return_value=pipeline), patch.object(sys, 'argv', ['query.py', 'Scope?', '--source-doc', 'sample']), redirect_stdout(io.StringIO()) as output:
            query.main()
        self.assertTrue(pipeline.called)
        self.assertIn('PDF page', output.getvalue())

    def test_cli_failed_one_shot_exits_nonzero(self):
        class Pipeline:
            def answer(self, *args, **kwargs):
                raise service.PipelineError('model failed')
        with patch.object(query, 'get_service', return_value=Pipeline()), patch.object(sys, 'argv', ['query.py', 'Scope?']), redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                query.main()
        self.assertEqual(raised.exception.code, 1)


class UITests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events = []
        self.messages = []
        self.session = {}
        outer = self
        fake = types.ModuleType('chainlit')
        fake.on_chat_start = lambda fn: fn
        fake.on_message = lambda fn: fn

        class Message:
            def __init__(self, content):
                self.content = content

            async def send(self):
                outer.messages.append(self.content)

            async def remove(self):
                outer.events.append('status_removed')

        class Step:
            def __init__(self, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                outer.events.append('step_sent')
        fake.Message, fake.Step = Message, Step
        fake.user_session = types.SimpleNamespace(
            get=self.session.get, set=lambda k, v: self.session.__setitem__(k, v))

        def make_async(fn):
            async def run(*args, **kwargs):
                return await asyncio.to_thread(fn, *args, **kwargs)
            return run
        fake.make_async = make_async
        self.module_patch = patch.dict(sys.modules, {'chainlit': fake})
        self.module_patch.start()
        spec = importlib.util.spec_from_file_location('tested_chainlit_app', Path(
            __file__).resolve().parents[1] / 'src' / 'chainlit_app.py')
        self.app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.app)

    async def asyncTearDown(self):
        self.module_patch.stop()

    async def test_startup_runs_in_worker_and_removes_status(self):
        main_thread = threading.get_ident()

        def initialize():
            self.assertNotEqual(threading.get_ident(), main_thread)
            return object()
        with patch.object(self.app, 'get_service', side_effect=initialize):
            await self.app.on_chat_start()
        self.assertIsNotNone(self.session['service'])
        self.assertIn('status_removed', self.events)

    async def test_model_error_surfaces_and_cleans_status(self):
        with patch.object(self.app, 'get_service', side_effect=RuntimeError('missing model')), self.assertLogs(self.app.log, level='ERROR'):
            await self.app.on_chat_start()
        self.assertTrue(any('missing model' in m for m in self.messages))
        self.assertIn('status_removed', self.events)
        self.assertIsNone(self.session['service'])

    async def test_answer_sources_use_async_step(self):
        class Pipeline:
            def answer(self, question):
                return service.Answer('Work [S1]', [dict(chunk(), source_id='S1', label='PMBOK, PDF page 3')])
        self.session['service'] = Pipeline()
        await self.app.on_message(types.SimpleNamespace(content='Question'))
        self.assertIn('step_sent', self.events)
        self.assertIn('PDF page 3', self.messages[-1])


@unittest.skipUnless(importlib.util.find_spec('chromadb'), 'Chroma unavailable; install requirements for real DB integration')
class RealChromaTests(TemporaryIndex):
    def test_real_db_snapshot_replace_and_filter(self):
        from storage import chroma_client, store_documents, load_collection
        from retrieval import HybridRetriever
        from test_pipeline import FakeBM25
        client = chroma_client()
        store_documents([chunk('Old document', source='a')], Embedder(), 'a', client)
        store_documents([chunk('Risk register', source='b')], Embedder(), 'b', client)
        store_documents([chunk('Revised scope', source='a')], Embedder(), 'a', client)
        collection = load_collection(client)
        self.assertEqual(set(collection.get()['documents']), {'Risk register', 'Revised scope'})
        hits = HybridRetriever(collection, FakeBM25).search('scope', Embedder(), source_doc='a')
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]['metadata']['source_doc'], 'a')


if __name__ == '__main__':
    unittest.main()
