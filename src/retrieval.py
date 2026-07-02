"""Document-filtered hybrid retrieval and identity-preserving rank fusion."""
import re
import threading

import config


_ATTRIBUTION_SUFFIX = re.compile(
    r'\s+(?:according to|as defined in)\s+[^?.!]+[?.!]*\s*$', re.I)
_ATTRIBUTION_PREFIX = re.compile(
    r'^\s*(?:according to|as defined in)\s+[^,;:]+[,;:]\s*', re.I)
_STOP_WORDS = set(
    'what is are a an the of in to according does do mean means define definition please explain'.split())
_FOOTER_LINE = re.compile(
    r'^\s*(?:A (?:Guide|Handbook|Manual) to .*'
    r'|[®]\s*(?:Guide|Handbook|Manual)\).*Edition'
    r'|(?:[©]|\(c\))\s*\d{4}.*'
    r'|(?:[A-Z]\s+)?NAVIGATI(?:ON)?(?: LINKS)?[^a-z]*)\s*$', re.I)


def focused_query(query):
    """Remove source-attribution wrappers from the search query."""
    result = _ATTRIBUTION_PREFIX.sub('', query.strip())
    result = _ATTRIBUTION_SUFFIX.sub('', result).strip()
    if result != query.strip() and result:
        return result.rstrip('?.!') + '?'
    return result


def clean_for_search(text):
    """Suppress known running-footer lines in ranking, never in stored evidence."""
    return '\n'.join(line for line in text.splitlines() if not _FOOTER_LINE.fullmatch(line))


def definition_term(query):
    question = focused_query(query).strip().rstrip('?.!')
    match = re.match(
        r'^(?:what (?:is|are)|define|(?:give (?:me )?)?(?:the )?definition of)\s+(.+)$', question, re.I)
    if not match:
        match = re.match(r'^what does (.+?) mean$', question, re.I)
    if not match:
        return None
    term = re.sub(r'^(?:a|an|the)\s+', '', match.group(1).strip(), flags=re.I)
    if not 1 <= len(term.split()) <= 8:
        return None
    return term


def comparison_terms(query):
    question = focused_query(query).strip().rstrip('?.!')
    patterns = (
        r'^how (?:does|do) (.+?) differ from (.+)$',
        r'^how (?:is|are) (.+?) different from (.+)$',
        r'^(?:what (?:is|are) (?:the )?)?differences? between (.+?) and (.+)$',
        r'^compare (.+?) (?:and|with|to|versus|vs\.?) (.+)$',
    )
    for pattern in patterns:
        match = re.match(pattern, question, re.I)
        if match:
            terms = tuple(re.sub(r'^(?:a|an|the)\s+', '', part.strip(), flags=re.I)
                          for part in match.groups())
            if all(1 <= len(term.split()) <= 8 for term in terms):
                return terms
    return ()


def definition_evidence(query, chunk):
    """Find literal entry paragraphs for concepts named in a comparison."""
    spans = []
    text = chunk['text']
    for term in comparison_terms(query):
        pattern = (r'(?im)^[ \t]*(?:[•*-][ \t]*)?' + _definition_label(term)
                   + r'(?:[ \t]*\[[^\]\n]{1,80}\])?[ \t]*[.:–—][ \t]*')
        for match in re.finditer(pattern, text):
            if not _entry_boundary_is_clear(text, match):
                continue
            body = _safe_definition_body(text[match.end():])
            if len(body) < 12 or re.match(r'^(?:see|refer to)\b', body, re.I):
                continue
            entry = ' '.join(match.group(0).split()) + ' ' + body
            if entry in ' '.join(text.split()):
                spans.append((term, entry))
                break
    return spans


def is_direct_definition(query, chunk):
    # Ranking and answer extraction must agree about what counts as a
    # complete definition. A heading mentioning an input is not enough.
    return extract_exact_definition(query, chunk) is not None


class HybridRetriever:
    def __init__(self, collection, bm25_factory=None):
        self.collection = collection
        self._bm25_factory = bm25_factory
        self._indexes = {}
        self._lock = threading.RLock()

    @staticmethod
    def _tokenize(text):
        return re.findall(r'\w+', text.lower())

    @classmethod
    def _lexical_tokens(cls, text):
        return [token for token in cls._tokenize(clean_for_search(text)) if token not in _STOP_WORDS]

    def _index(self, source_doc):
        with self._lock:
            if source_doc in self._indexes:
                return self._indexes[source_doc]
            kwargs = {'where': {'source_doc': source_doc}} if source_doc else {}
            result = self.collection.get(include=['documents', 'metadatas'], **kwargs)
            tokens = [self._lexical_tokens(t) for t in result['documents']]
            bm25 = None
            if tokens and any(tokens):
                factory = self._bm25_factory
                if factory is None:
                    from rank_bm25 import BM25Okapi
                    factory = BM25Okapi
                bm25 = factory(tokens)
            self._indexes[source_doc] = result, bm25
            return result, bm25

    def prepare(self, source_doc=None):
        """Build the lexical index before accepting queries."""
        self._index(source_doc)

    def metadata_for(self, ids, source_doc=None):
        corpus, _ = self._index(source_doc)
        requested = set(ids)
        return {chunk_id: dict(id=chunk_id, metadata=metadata)
                for chunk_id, metadata in zip(corpus['ids'], corpus['metadatas'])
                if chunk_id in requested}

    def search(self, query, embedding_model, top_k=10, source_doc=None):
        if not query.strip() or top_k <= 0:
            return []
        corpus, bm25 = self._index(source_doc)
        count = len(corpus['ids'])
        if not count:
            return []
        search_query = focused_query(query)
        embedding = embedding_model.encode([search_query], normalize_embeddings=True)
        embedding = embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        kwargs = {'where': {'source_doc': source_doc}} if source_doc else {}
        vector = self.collection.query(query_embeddings=embedding,
                                       n_results=min(top_k, count), include=['documents', 'metadatas', 'distances'], **kwargs)
        merged = {}
        for rank, (chunk_id, text, meta, distance) in enumerate(zip(
                vector['ids'][0], vector['documents'][0], vector['metadatas'][0], vector['distances'][0]), 1):
            merged[chunk_id] = dict(id=chunk_id, text=text, metadata=meta,
                                    vector_distance=float(distance), bm25_score=0.0, fusion_score=1 / (60 + rank))
        if bm25:
            tokens = self._lexical_tokens(search_query)
            scores = bm25.get_scores(tokens)
            # Negative IDFs can occur in tiny corpora; require token overlap,
            # rather than assuming all legitimate BM25 hits have positive scores.
            matching = [i for i, text in enumerate(corpus['documents'])
                        if set(tokens).intersection(self._lexical_tokens(text))]
            matching.sort(key=lambda i: scores[i], reverse=True)
            for rank, idx in enumerate(matching[:top_k], 1):
                chunk_id = corpus['ids'][idx]
                chunk = merged.setdefault(chunk_id, dict(id=chunk_id,
                                                         text=corpus['documents'][idx], metadata=corpus['metadatas'][idx],
                                                         fusion_score=0.0, vector_distance=None))
                chunk['bm25_score'] = float(scores[idx])
                chunk['fusion_score'] += 1 / (60 + rank)
        # Exact definitional entries are drawn from the existing corpus too,
        # so a noisy vector/BM25 top-k cannot hide them from the cross-encoder.
        # This supplements candidates only; it does not synthesize an answer.
        concepts = (definition_term(search_query),) if definition_term(
            search_query) else comparison_terms(search_query)
        for concept in concepts:
            added = 0
            for idx, text in enumerate(corpus['documents']):
                candidate = dict(id=corpus['ids'][idx], text=text,
                                 metadata=corpus['metadatas'][idx])
                if is_direct_definition('What is ' + concept + '?', candidate):
                    candidate.update(fusion_score=1 / 61, vector_distance=None, bm25_score=0.0,
                                     candidate_origin='exact_definition')
                    merged.setdefault(candidate['id'], candidate)
                    added += 1
                    if added >= 6:
                        break
        return sorted(merged.values(), key=lambda c: c['fusion_score'], reverse=True)


class Reranker:
    def __init__(self, model=None):
        self.model = model
        self._lock = threading.RLock()

    def preload(self):
        with self._lock:
            if self.model is None:
                from sentence_transformers import CrossEncoder
                kwargs = {'revision': config.CROSS_ENCODER_REVISION} if config.CROSS_ENCODER_REVISION else {}
                self.model = CrossEncoder(config.CROSS_ENCODER_MODEL, device='cpu', **kwargs)
        return self

    def rerank(self, query, chunks, top_k=3):
        if not chunks or top_k <= 0:
            return []
        self.preload()
        with self._lock:
            scores = self.model.predict(
                [(focused_query(query), clean_for_search(c['text'])) for c in chunks])
        scored = [dict(c, rerank_score=float(score),
                       definition_match=is_direct_definition(query, c),
                       comparison_coverage=len(definition_evidence(query, c))) for c, score in zip(chunks, scores)]
        # For direct definition questions, prefer exact term entries that also
        # pass the normal relevance threshold. Preserve raw scores unchanged.

        def key(chunk):
            relevant = chunk['rerank_score'] >= config.MIN_RERANK_SCORE
            return (chunk['comparison_coverage'] if relevant else 0,
                    chunk['definition_match'] and relevant, chunk['rerank_score'])
        return sorted(scored, key=key, reverse=True)[:top_k]


def _canonical_term(text):
    # An uppercase parenthetical is an alias, not a different concept.
    text = re.sub(r'\((?-i:[A-Z][A-Z0-9/&. -]{0,24})\)', '', text)
    return ' '.join(re.findall(r'\w+', text.casefold()))


def _definition_label(term):
    words = r'\s+'.join(re.escape(word) for word in term.split())
    acronym = r'(?:[ \t]*\((?-i:[A-Z][A-Z0-9/&. -]{0,24})\))?'
    direct = words + acronym
    # Resolve short-form questions against full entry headings with aliases.
    if re.fullmatch(r'[A-Za-z][A-Za-z0-9/&.-]{1,14}', term):
        alias = r'[A-Za-z][A-Za-z /&-]{1,90}[ \t]+\(' + re.escape(term) + r'\)'
        return '(?:' + direct + '|' + alias + ')'
    return direct


def _safe_definition_body(body):
    body = body.lstrip()
    body = re.split(r'\n[ \t]*\n', body, maxsplit=1)[0]
    body = re.split(
        r'\n[ \t]*(?:[•*-][ \t]*)?[A-Z][A-Za-z0-9 /()&-]{1,90}'
        r'(?:[ \t]*\[[^\]\n]+\])?[.:][ \t]+(?=(?:The|A|An|See|Refer)\b)',
        body, maxsplit=1)[0]
    return ' '.join(body.split())


def _entry_boundary_is_clear(text, match):
    """Reject lowercase/wrapped running text masquerading as an entry label."""
    first_letter = next((char for char in match.group(0) if char.isalpha()), '')
    if not first_letter or not first_letter.isupper():
        return False
    if match.start() == 0:
        return True
    prefix = text[:match.start()]
    # A blank paragraph or bullet separates a heading from preceding prose.
    if re.search(r'\n[ \t]*\n[ \t]*$', prefix):
        return True
    previous = prefix.rstrip().split('\n')[-1].strip() if prefix.strip() else ''
    if not previous or previous in {'•', '-', '*'} or re.fullmatch(r'\.?\d+(?:\.\d+)*\.?', previous):
        return True
    # A line ending mid-sentence is a PDF line wrap, not a new glossary entry.
    return previous[-1] in '.!?:;'


def extract_exact_definition(query, chunk):
    """Return a complete original definition span or None; never generate one."""
    term = definition_term(query)
    if term is None:
        return None
    original = chunk['text']
    normalized = ' '.join(original.split())
    label = _definition_label(term)
    start = (r'(?im)^[ \t]*(?:[•*-][ \t]*)?(?:\.?\d+(?:\.\d+)*\.?[ \t]+)?'
             + label + r'(?:[ \t]*\[[^\]\n]{1,80}\])?')
    patterns = (
        (start + r'[ \t]*[.:–—][ \t]*', True),
        (start + r'[ \t]*\n', False),
    )
    for pattern, formal in patterns:
        for match in re.finditer(pattern, original):
            if not _entry_boundary_is_clear(original, match):
                continue
            body = _safe_definition_body(original[match.end():])
            if not 20 <= len(body) <= 600 or body[-1] not in '.!?':
                continue
            if re.match(r'^(?:see|refer to|figure|table)\b', body, re.I):
                continue
            if not formal:
                subject = re.sub(r'^the\s+', '', body, flags=re.I)
                if re.match(re.escape(term) + r'\b', subject, re.I):
                    continue
                # Bare headings need a definitional lead-in rather than a usage description.
                if not re.match(r'^(?:an?\s+|the\s+(?:work|process|sum|set|collection|features|functions)\b)', body, re.I):
                    continue
            definition = ' '.join(match.group(0).split()) + ' ' + body
            if definition in normalized:
                return definition
    meta = chunk.get('metadata', {})
    if (meta.get('type') == 'glossary' and meta.get('term')
            and _canonical_term(meta['term']) == _canonical_term(term)):
        body = _safe_definition_body(original)
        if 20 <= len(body) <= 600 and body[-1] in '.!?' and not re.match(r'^(?:see|refer to)\b', body, re.I):
            return body if body in normalized else None
    return None
