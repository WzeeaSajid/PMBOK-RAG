"""Publish immutable Chroma snapshots through an atomic local manifest.

Existing readers keep their consistent snapshot. Reload them after ingestion.
"""
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

import config


class IndexErrorBase(RuntimeError):
    pass


def chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


def read_state():
    path = Path(config.INDEX_STATE_FILE)
    if not path.exists():
        return None
    try:
        state = json.loads(path.read_text(encoding='utf-8'))
        if (not isinstance(state, dict) or not isinstance(state.get('collection'), str)
                or not state['collection'] or not isinstance(state.get('signature'), dict)):
            raise ValueError('Invalid index manifest structure.')
        return state
    except (ValueError, OSError) as exc:
        raise IndexErrorBase(
            'Index manifest is unreadable; restore it or rebuild in a new PMBOK_CHROMA_DIR.') from exc


def validate_state(state):
    if state.get('signature') != config.index_signature():
        raise IndexErrorBase(
            'Index model/revision/chunking/schema mismatch. Re-ingest all PDFs into a new PMBOK_CHROMA_DIR.')


def load_collection(client=None):
    state = read_state()
    if state is None:
        raise IndexErrorBase(
            'No compatible index. Run python src/ingest.py <PDF>. Legacy indexes need re-ingestion to repair provenance.')
    validate_state(state)
    return (client or chroma_client()).get_collection(state['collection'])


@contextmanager
def writer_lock():
    directory = Path(config.CHROMA_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / 'ingest.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise IndexErrorBase(
            'Another ingestion holds ingest.lock. After a crash, verify that process has stopped before removing the lock.') from exc
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def _add_batches(collection, records, batch_size=128):
    for start in range(0, len(records['ids']), batch_size):
        collection.add(**{k: values[start:start + batch_size] for k, values in records.items()})


def store_documents(docs, embedding_model, source_doc, client=None):
    """Replace one document; publish only after all embedding/storage succeeds."""
    if not docs:
        raise ValueError('No usable content extracted; existing index left intact.')
    if any(d['metadata'].get('source_doc') != source_doc for d in docs):
        raise ValueError('All replacement chunks must belong to source_doc.')
    if len({d['id'] for d in docs}) != len(docs):
        raise ValueError('Chunk IDs must be unique within a document.')
    client = client or chroma_client()
    with writer_lock():
        state = read_state()
        if state:
            validate_state(state)
        name = f'{config.COLLECTION_NAME}_{uuid.uuid4().hex}'
        staging = client.create_collection(
            name, metadata={'hnsw:space': config.CHROMA_DISTANCE_METRIC})
        temporary = Path(config.INDEX_STATE_FILE + '.tmp')
        published = False
        try:
            retained = 0
            if state:
                previous = client.get_collection(state['collection'])
                for offset in range(0, previous.count(), 128):
                    batch = previous.get(limit=128, offset=offset, include=[
                                         'documents', 'metadatas', 'embeddings'])
                    keep = [i for i, m in enumerate(
                        batch['metadatas']) if m['source_doc'] != source_doc]
                    if keep:
                        records = {key: [batch[key][i] for i in keep]
                                   for key in ('ids', 'documents', 'metadatas', 'embeddings')}
                        records['embeddings'] = [list(e) for e in records['embeddings']]
                        _add_batches(staging, records)
                        retained += len(keep)
            for start in range(0, len(docs), 32):
                batch = docs[start:start + 32]
                texts = [d['text'] for d in batch]
                embeddings = embedding_model.encode(texts, batch_size=32, normalize_embeddings=True)
                embeddings = embeddings.tolist() if hasattr(embeddings, 'tolist') else embeddings
                _add_batches(staging, dict(ids=[d['id'] for d in batch], documents=texts,
                                           metadatas=[d['metadata'] for d in batch], embeddings=embeddings))
            if staging.count() != retained + len(docs):
                raise IndexErrorBase('Staged chunk count mismatch; active index unchanged.')
            with temporary.open('w', encoding='utf-8') as stream:
                json.dump({'collection': name, 'signature': config.index_signature()}, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, config.INDEX_STATE_FILE)
            published = True
            return name
        finally:
            temporary.unlink(missing_ok=True)
            if not published:
                client.delete_collection(name)
