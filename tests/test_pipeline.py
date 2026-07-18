"""Offline regression tests. Model/DB/network boundaries are explicit doubles."""
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import _bootstrap

import config
import ingest
import retrieval
import service
import storage


class Embedder:
    def encode(self, texts, **kwargs):
        return [[float(len(text)), 1.0] for text in texts]


class FakeBM25:
    def __init__(self, corpus):
        self.corpus = corpus

    def get_scores(self, query):
        return [sum(token in text for token in query) for text in self.corpus]


class Collection:
    def __init__(self, name='fake'):
        self.name = name
        self.rows = {}
        self.calls = []

    def count(self):
        return len(self.rows)

    def add(self, ids, documents, metadatas, embeddings):
        for i, text, meta, embedding in zip(ids, documents, metadatas, embeddings):
            if i in self.rows:
                raise ValueError('duplicate ID')
            self.rows[i] = dict(id=i, document=text, metadata=meta, embedding=embedding)

    def get(self, include=None, where=None, offset=0, limit=None):
        self.calls.append(('get', where))
        rows = [r for r in self.rows.values() if not where or all(
            r['metadata'].get(k) == v for k, v in where.items())]
        rows = rows[offset:offset + limit if limit is not None else None]
        return {key: [r[field] for r in rows] for key, field in (
            ('ids', 'id'), ('documents', 'document'), ('metadatas', 'metadata'), ('embeddings', 'embedding'))}

    def query(self, query_embeddings, n_results, include, where=None):
        self.calls.append(('query', where))
        results = self.get(where=where, limit=n_results)
        results = {key: [value] for key, value in results.items() if key != 'embeddings'}
        results['distances'] = [[0.1] * len(results['ids'][0])]
        return results


class Client:
    def __init__(self):
        self.collections = {}

    def create_collection(self, name, **kwargs):
        self.collections[name] = Collection(name)
        return self.collections[name]

    def get_collection(self, name):
        return self.collections[name]

    def delete_collection(self, name):
        del self.collections[name]


def chunk(text='Scope is the work needed to deliver the product.', source='pmbok', page=3, kind='text'):
    return dict(id=f'{source}-{page}-{text}', text=text, metadata=dict(source_doc=source,
                                                                       document_title=source.title(), edition='3rd', page_number=page,
                                                                       page_label=str(page), type=kind))


class TemporaryIndex(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.config_patch = patch.multiple(config, CHROMA_DIR=str(self.directory / 'db'),
                                           INDEX_STATE_FILE=str(
                                               self.directory / 'db' / 'active_index.json'),
                                           IMAGES_DIR=str(self.directory / 'images'),
                                           CAPTION_CACHE_FILE=str(self.directory / 'data' / 'caption_cache.json'))
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.temp.cleanup()


class ChunkTests(unittest.TestCase):
    def test_long_paragraph_is_bounded(self):
        for text in ('x' * 2000, 'word ' * 1000, '管理范围 ' * 1000):
            parts = list(ingest.split_with_offsets(text, 80, 10))
            self.assertTrue(
                all(0 < len(part) <= 80 and text[start:end] == part for start, end, part in parts))
            covered = set(i for start, end, _ in parts for i in range(start, end))
            self.assertTrue(all(i in covered for i, char in enumerate(text) if not char.isspace()))

    def test_bad_chunk_configuration(self):
        for maximum, overlap in ((0, 0), (80, 80), (80, -1)):
            with self.assertRaises(ValueError):
                ingest.chunk_text('abc', maximum, overlap)

    def test_empty_text_has_no_chunks(self):
        self.assertEqual(ingest.chunk_text('   ', 80, 10), [])

    def test_page_labels_do_not_drift(self):
        elements = [dict(type='body', text=f'PAGE{i} ' + 'word ' *
                         30, page=i, page_label=str(i + 10)) for i in range(1, 9)]
        with patch.multiple(config, CHUNK_MAX_CHARACTERS=80, CHUNK_OVERLAP=20):
            docs = ingest.build_body_chunks(elements, 'test')
        for doc in docs:
            meta = doc['metadata']
            self.assertEqual(meta['page_start'], meta['page_end'])
            self.assertEqual(meta['page_label'], str(meta['page_number'] + 10))
            original = elements[meta['page_number'] - 1]['text'].strip()
            self.assertEqual(original[meta['char_start']:meta['char_end']], doc['text'])
        self.assertEqual({d['metadata']['page_number'] for d in docs}, set(range(1, 9)))

    def test_sections_and_trailing_headings_survive(self):
        elements = [dict(type='title', text='Scope', page=1), dict(type='body', text='Scope details', page=1),
                    dict(type='title', text='Risk', page=1), dict(type='title', text='Final heading', page=2)]
        docs = ingest.build_body_chunks(elements, 'test')
        self.assertEqual([d['metadata']['section']
                         for d in docs], ['Scope', 'Risk', 'Final heading'])
        self.assertEqual(docs[-1]['text'], 'Final heading')

    def test_cpi_and_definitions_are_not_backmatter(self):
        for title in ('Cost Performance Index', 'Schedule Performance Index', 'Activity Definitions', 'Definitions of Project Scope'):
            streams = ingest.split_streams([dict(type='title', text=title, page=1),
                                           dict(type='body', text='Useful content', page=1)])
            self.assertEqual(len(streams['body']), 2, title)
            self.assertFalse(streams['index'])

    def test_actual_backmatter_is_excluded(self):
        streams = ingest.split_streams([dict(type='title', text='INDEX', page=1),
                                        dict(type='body', text='A 1, 2, 3', page=1), dict(
                                            type='title', text='Appendix A', page=2),
                                        dict(type='body', text='Retain appendix', page=2)])
        self.assertEqual(len(streams['index']), 2)
        self.assertEqual(len(streams['body']), 2)

    def test_glossary_letter_sections_and_size(self):
        streams = ingest.split_streams([dict(type='title', text='Glossary', page=10),
                                        dict(type='title', text='A', page=10), dict(type='body', text='Activity: ' + 'explanation ' * 100, page=10)])
        docs = ingest.build_glossary_docs(
            ingest.parse_glossary_entries(streams['glossary']), 'test')
        self.assertTrue(all(len(d['text']) <= config.CHUNK_MAX_CHARACTERS for d in docs))
        self.assertTrue(any('Activity' in d['text'] for d in docs))


class StorageTests(TemporaryIndex):
    def test_replace_does_not_retain_stale_text_and_preserves_other_docs(self):
        client, embed = Client(), Embedder()
        storage.store_documents([chunk('Old scope passage', source='a')], embed, 'a', client)
        storage.store_documents([chunk('Risk document', source='b')], embed, 'b', client)
        previous = storage.load_collection(client)
        storage.store_documents([chunk('Revised scope passage', source='a')], embed, 'a', client)
        active = storage.load_collection(client)
        self.assertEqual(set(active.get()['documents']), {'Revised scope passage', 'Risk document'})
        # Existing reader remains consistent.
        self.assertIn('Old scope passage', previous.get()['documents'])

    def test_failed_embedding_keeps_active_snapshot(self):
        client = Client()
        storage.store_documents([chunk('Old passage')], Embedder(), 'pmbok', client)
        before = Path(config.INDEX_STATE_FILE).read_bytes()

        class Broken:
            def encode(self, *args, **kwargs):
                raise RuntimeError('embedding failed')
        with self.assertRaises(RuntimeError):
            storage.store_documents([chunk('New passage')], Broken(), 'pmbok', client)
        self.assertEqual(before, Path(config.INDEX_STATE_FILE).read_bytes())
        self.assertEqual(len(client.collections), 1)
        self.assertFalse((Path(config.CHROMA_DIR) / 'ingest.lock').exists())

    def test_failed_publication_keeps_active_snapshot(self):
        client = Client()
        storage.store_documents([chunk('Old passage')], Embedder(), 'pmbok', client)
        before = storage.read_state()
        with patch.object(storage.os, 'replace', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError):
                storage.store_documents([chunk('New passage')], Embedder(), 'pmbok', client)
        self.assertEqual(before, storage.read_state())
        self.assertEqual(len(client.collections), 1)

    def test_empty_replacement_and_mixed_sources_rejected(self):
        with self.assertRaises(ValueError):
            storage.store_documents([], Embedder(), 'a', Client())
        with self.assertRaises(ValueError):
            storage.store_documents([chunk(source='b')], Embedder(), 'a', Client())

    def test_signature_mismatch_is_detected(self):
        client = Client()
        storage.store_documents([chunk()], Embedder(), 'pmbok', client)
        for setting in ('EMBEDDING_MODEL_NAME', 'EMBEDDING_MODEL_REVISION', 'CHUNKING_VERSION'):
            with patch.object(config, setting, 'changed'):
                with self.assertRaises(storage.IndexErrorBase):
                    storage.load_collection(client)

    def test_second_writer_is_rejected(self):
        with storage.writer_lock():
            with self.assertRaises(storage.IndexErrorBase):
                with storage.writer_lock():
                    self.fail('second writer entered')

    def test_large_snapshot_is_batched(self):
        client = Client()
        storage.store_documents([chunk('Old doc ' + str(i), source='b', page=i)
                                for i in range(260)], Embedder(), 'b', client)
        storage.store_documents([chunk('New doc', source='a')], Embedder(), 'a', client)
        self.assertEqual(storage.load_collection(client).count(), 261)


class RetrievalTests(unittest.TestCase):
    def collection(self, chunks):
        collection = Collection()
        collection.add(ids=[c['id'] for c in chunks], documents=[c['text'] for c in chunks],
                       metadatas=[c['metadata'] for c in chunks], embeddings=[[0., 1.]] * len(chunks))
        return collection

    def test_distinct_same_prefix_chunks_survive(self):
        collection = self.collection(
            [chunk('A' * 100 + ' scope', page=1), chunk('A' * 100 + ' risk', page=2)])
        result = retrieval.HybridRetriever(collection, FakeBM25).search('scope', Embedder())
        self.assertEqual(len(result), 2)
        self.assertTrue(all('fusion_score' in c for c in result))

    def test_both_retrievers_filter_before_ranking(self):
        collection = self.collection([chunk(source='other', page=i)
                                     for i in range(20)] + [chunk(source='target')])
        result = retrieval.HybridRetriever(collection, FakeBM25).search(
            'scope', Embedder(), source_doc='target')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['metadata']['source_doc'], 'target')
        self.assertIn(('query', {'source_doc': 'target'}), collection.calls)
        self.assertIn(('get', {'source_doc': 'target'}), collection.calls)

    def test_empty_collection_never_embeds_or_initializes_bm25(self):
        def forbidden(*args):
            self.fail('constructed BM25 on empty corpus')
        result = retrieval.HybridRetriever(Collection(), forbidden).search('q', None)
        self.assertEqual(result, [])

    def test_zero_match_bm25_does_not_add_arbitrary_chunks(self):
        collection = self.collection(
            [chunk('scope', page=1), chunk('risk', page=2), chunk('cost', page=3)])
        result = retrieval.HybridRetriever(collection, FakeBM25).search(
            'astronomy', Embedder(), top_k=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['bm25_score'], 0)

    def test_empty_reranking_does_not_load_model(self):
        self.assertEqual(retrieval.Reranker().rerank('q', []), [])

    def test_reranker_retains_scores(self):
        class Model:
            def predict(self, pairs):
                return [-4, 5]
        ranked = retrieval.Reranker(Model()).rerank('q', [chunk(page=1), chunk(page=2)])
        self.assertEqual(ranked[0]['metadata']['page_number'], 2)
        self.assertEqual(ranked[0]['rerank_score'], 5.)


class ServiceTests(unittest.TestCase):
    def make_service(self, raw=None, score=5):
        passage = chunk()

        class Retriever:
            def search(self, *args, **kwargs):
                self.last_kwargs = kwargs
                return [passage]

        class Reranker:
            def rerank(self, query, chunks, **kwargs):
                return [dict(c, rerank_score=score) for c in chunks]

        class LLM:
            calls = 0

            def generate(self, messages):
                self.calls += 1
                self.messages = messages
                return raw or json.dumps(dict(unanswerable=False, statements=[dict(
                    text='Scope describes required work.', source_ids=['S1'],
                    quotes=[dict(source_id='S1', text='Scope is the work needed')])]))
        return service.RAGService(Embedder(), Retriever(), Reranker(), LLM())

    def test_shared_service_produces_validated_sources(self):
        pipeline = self.make_service()
        answer = pipeline.answer('What is scope?', source_doc='pmbok')
        self.assertFalse(answer.refused)
        self.assertIn('[S1]', answer.answer)
        self.assertEqual(answer.sources[0]['label'], 'Pmbok (3rd), PDF page 3')
        self.assertEqual(pipeline.retriever.last_kwargs['source_doc'], 'pmbok')
        self.assertEqual(pipeline.llm.messages[0]['role'], 'system')

    def test_low_relevance_refuses_without_generation(self):
        pipeline = self.make_service(score=-20)
        self.assertTrue(pipeline.answer('Who won the football match?').refused)
        self.assertEqual(pipeline.llm.calls, 0)

    def test_unknown_citation_refuses(self):
        raw = json.dumps(dict(unanswerable=False, statements=[
                         dict(text='Scope.', source_ids=['S999'], quotes=[])]))
        answer = self.make_service(raw).answer('Scope?')
        self.assertTrue(answer.refused)
        self.assertEqual(answer.diagnostics['reason'], 'unknown_source')

    def test_inline_fabricated_citation_refuses(self):
        for text in ('Scope is work [S999]', 'Scope appears on page 999'):
            raw = json.dumps(dict(unanswerable=False, statements=[dict(text=text, source_ids=['S1'],
                                                                       quotes=[dict(source_id='S1', text='Scope is the work needed')])]))
            answer = self.make_service(raw).answer('Scope?')
            self.assertTrue(answer.refused)
            self.assertEqual(answer.diagnostics['reason'], 'inline_citation')

    def test_invented_quote_refuses(self):
        raw = json.dumps(dict(unanswerable=False, statements=[dict(text='Scope.', source_ids=['S1'],
                                                                   quotes=[dict(source_id='S1', text='This text does not exist in the document')])]))
        answer = self.make_service(raw).answer('Scope?')
        self.assertTrue(answer.refused)
        self.assertEqual(answer.diagnostics['reason'], 'quote_not_in_source')

    def test_every_cited_source_requires_own_quote(self):
        chunks, _ = service.select_context([chunk(page=1), chunk(page=2)])
        raw = json.dumps(dict(unanswerable=False, statements=[dict(text='Scope.', source_ids=['S1', 'S2'],
                                                                   quotes=[dict(source_id='S1', text='Scope is the work needed')])]))
        self.assertEqual(service.validate_answer(raw, chunks)[1], 'missing_supporting_quote')

    def test_malformed_or_empty_answers_refuse(self):
        for raw in ('not json', '[]', '{}', '{"unanswerable":false,"statements":[]}',
                    '{"unanswerable":true,"statements":[]}'):
            self.assertTrue(self.make_service(raw).answer('Question').refused)

    def test_whitespace_normalized_quote(self):
        chunks, _ = service.select_context([chunk()])
        raw = json.dumps(dict(unanswerable=False, statements=[dict(text='Scope.', source_ids=['S1'],
                                                                   quotes=[dict(source_id='S1', text='Scope is\n the work needed')])]))
        self.assertEqual(service.validate_answer(raw, chunks)[1], 'validated')

    def test_complete_passages_not_truncated(self):
        text = 'x' * 800 + ' ESSENTIAL ANSWER'
        selected, records = service.select_context([chunk(text)], token_budget=2000)
        self.assertEqual(selected[0]['text'], text)
        self.assertIn('ESSENTIAL ANSWER', service.build_messages('q', records)[1]['content'])

    def test_context_budget_includes_json_and_unicode(self):
        selected, records = service.select_context(
            [chunk('管理 ' * 50, page=i) for i in range(10)], token_budget=1800)
        self.assertLessEqual(len(json.dumps(records, ensure_ascii=False).encode()), 1800)
        self.assertEqual([c['source_id'] for c in selected], [
                         f'S{i + 1}' for i in range(len(selected))])

    def test_invalid_questions(self):
        pipeline = self.make_service()
        for question in (' ', 'x' * (config.MAX_QUESTION_CHARACTERS + 1)):
            with self.assertRaises(ValueError):
                pipeline.answer(question)

    def test_labels_distinguish_pdf_and_printed_page(self):
        self.assertIn('printed page ix', service.source_label(
            dict(source_doc='pmbok', page_number=9, page_label='ix')))

    def test_ollama_model_preflight_checks_actual_model(self):
        class Client(service.OllamaClient):
            def request(self, *args, **kwargs):
                return {'models': [{'name': 'unrelated:latest'}]}
        with self.assertRaisesRegex(service.PipelineError, 'not installed'):
            Client().preflight()

    def test_ollama_preload_http_error_is_not_swallowed(self):
        def transport(request, timeout):
            if request.full_url.endswith('/api/tags'):
                return io.BytesIO(json.dumps({'models': [{'name': config.LLM_MODEL}]}).encode())
            raise HTTPError(request.full_url, 500, 'bad', {}, None)
        with self.assertRaisesRegex(service.PipelineError, 'HTTP 500'):
            service.OllamaClient(transport).preflight()

    def test_ollama_invalid_response_and_empty_output(self):
        with self.assertRaises(service.PipelineError):
            service.OllamaClient(lambda *a, **k: io.BytesIO(b'bad')).request('/api/tags')
        with self.assertRaises(service.PipelineError):
            service.OllamaClient(
                lambda *a, **k: io.BytesIO(b'{"message":{"content":""}}')).generate([])


class OCRTests(TemporaryIndex):
    def test_failed_ocr_is_not_cached_or_indexed(self):
        path = self.directory / 'sample.png'
        path.write_bytes(b'fake image')
        element = dict(type='image', page=4, image_path=str(path))

        def fail(path):
            raise RuntimeError('OCR unavailable')
        self.assertEqual(ingest.process_images([element], 'doc', ocr_fn=fail), [])
        self.assertEqual(ingest.load_caption_cache(), {})
        docs = ingest.process_images([element], 'doc', ocr_fn=lambda p: 'Real figure label')
        self.assertEqual(docs[0]['text'], 'Real figure label')
        self.assertTrue(ingest.load_caption_cache())

    def test_successful_cache_avoids_repeated_ocr(self):
        path = self.directory / 'image.png'
        path.write_bytes(b'image')
        element = dict(type='image', page=1, image_path=str(path))
        ingest.process_images([element], 'doc', ocr_fn=lambda p: 'Figure scope label')

        def unexpected(path):
            self.fail('cache missed')
        self.assertTrue(ingest.process_images([element], 'doc', ocr_fn=unexpected))

    def test_cache_recovers_from_invalid_json(self):
        path = Path(config.CAPTION_CACHE_FILE)
        path.parent.mkdir()
        path.write_text('{not json')
        self.assertEqual(ingest.load_caption_cache(), {})


@unittest.skipUnless(importlib.util.find_spec('fitz'), 'PyMuPDF unavailable')
class PDFIntegrationTests(TemporaryIndex):
    def make_pdf(self, path, title='Cost Performance Index'):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 70), title, fontsize=16)
        page.insert_text((50, 110), 'CPI is earned value divided by actual cost.', fontsize=11)
        doc.save(path)
        doc.close()

    def test_real_pdf_heading_and_cpi_extraction(self):
        path = self.directory / 'sample.pdf'
        self.make_pdf(path)
        elements = ingest.extract_pdf(str(path), render_figures=False)
        streams = ingest.split_streams(elements)
        docs = ingest.build_body_chunks(streams['body'], 'sample')
        self.assertTrue(any('earned value' in d['text'] for d in docs))
        self.assertFalse(streams['index'])
        self.assertTrue(all(d['metadata']['page_number'] == 1 for d in docs))

    def test_real_pdf_table_rows(self):
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        for x in (50, 200, 350):
            page.draw_line((x, 50), (x, 150))
        for y in (50, 100, 150):
            page.draw_line((50, y), (350, y))
        for position, text in (((60, 80), 'Metric'), ((210, 80), 'Formula'),
                               ((60, 130), 'CPI'), ((210, 130), 'EV / AC')):
            page.insert_text(position, text, fontsize=11)
        path = self.directory / 'table.pdf'
        doc.save(path)
        doc.close()
        tables = [el for el in ingest.extract_pdf(
            str(path), render_figures=False) if el['type'] == 'table']
        self.assertTrue(tables)
        self.assertIn('CPI | EV / AC', tables[0]['text'])

    def test_image_paths_are_document_specific(self):
        first, second = self.directory / 'a.pdf', self.directory / 'b.pdf'
        self.make_pdf(first, 'Cost diagram')
        self.make_pdf(second, 'Risk diagram')
        # Vector drawing triggers full-page render fallback.
        import fitz
        for path in (first, second):
            with fitz.open(path) as doc:
                doc[0].draw_rect(fitz.Rect(40, 40, 400, 160))
                doc.saveIncr()
        images_a = [e['image_path'] for e in ingest.extract_pdf(str(first)) if e['type'] == 'image']
        images_b = [e['image_path']
                    for e in ingest.extract_pdf(str(second)) if e['type'] == 'image']
        self.assertTrue(images_a and images_b)
        self.assertNotEqual(images_a[0], images_b[0])

    @unittest.skipUnless(shutil.which('tesseract'), 'Tesseract unavailable')
    def test_real_scanned_pdf_ocr(self):
        import fitz
        native = fitz.open()
        page = native.new_page(width=700, height=300)
        page.insert_text((40, 100), 'PROJECT SCOPE MANAGEMENT', fontsize=24)
        raster = page.get_pixmap(dpi=150).tobytes('png')
        scanned = fitz.open()
        page = scanned.new_page(width=700, height=300)
        page.insert_image(page.rect, stream=raster)
        path = self.directory / 'scan.pdf'
        scanned.save(path)
        native.close()
        scanned.close()
        elements = ingest.extract_pdf(str(path))
        images = [e for e in elements if e['type'] == 'image']
        self.assertEqual(images[0]['extraction'], 'rendered_page_ocr')
        docs = ingest.process_images(images, 'scan')
        self.assertTrue(any('PROJECT SCOPE MANAGEMENT' in d['text'].upper() for d in docs))


if __name__ == '__main__':
    unittest.main()
