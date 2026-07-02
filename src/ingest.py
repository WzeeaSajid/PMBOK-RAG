"""PDF extraction, bounded provenance-aware chunking, OCR, and safe indexing."""
import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import config
from storage import store_documents

log = logging.getLogger(__name__)
GLOSSARY_TITLE_RE = re.compile(
    r'^\s*(?:\d+[.\s]+)?(?:glossary(?: of terms)?|key terms|terminology|definitions)\s*$', re.I)
INDEX_TITLE_RE = re.compile(r'^\s*(?:subject |author )?index\s*$', re.I)
TOC_TITLE_RE = re.compile(r'^\s*(?:table\s+of\s+)?contents\s*$', re.I)
GLOSSARY_ENTRY_RE = re.compile(r"^\s*([A-Za-z0-9 ,'/()-]{1,80}?)\s*[:\u2013\u2014]\s+(.+)$", re.S)


def split_with_offsets(text, max_chars, overlap):
    """Yield exact (start, end, text) spans, bounded even without whitespace."""
    if max_chars <= 0 or not 0 <= overlap < max_chars:
        raise ValueError('Require chunk_size > 0 and 0 <= overlap < chunk_size.')
    start = 0
    while start < len(text):
        limit = min(start + max_chars, len(text))
        end = limit
        if limit < len(text):
            # Prefer paragraph, sentence, then word boundaries in the latter half.
            floor = start + max_chars // 2
            for pattern in (r'\n\n', r'(?<=[.!?])\s+', r'\s+'):
                boundaries = list(re.finditer(pattern, text[floor:limit]))
                if boundaries:
                    end = floor + boundaries[-1].end()
                    break
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            yield left, right, text[left:right]
        if end == len(text):
            break
        next_start = max(start + 1, end - overlap)
        # Start overlapping chunks at a word boundary where possible.
        if overlap and next_start > 0 and not text[next_start - 1].isspace():
            boundary = re.search(r'\s+', text[next_start:end])
            if boundary:
                next_start += boundary.end()
        start = next_start


def chunk_text(text, max_chars, overlap):
    return [part for _, _, part in split_with_offsets(text, max_chars, overlap)]


def _heading(block):
    text = block['text']
    spans = block['spans']
    return (len(text) <= 200 and len(text.splitlines()) <= 3 and bool(spans)
            and (max(s.get('size', 0) for s in spans) >= 14
                 or all(s.get('flags', 0) & 16 and s.get('size', 0) >= 12 for s in spans)))


def extract_pdf(pdf_path, render_figures=True):
    import fitz
    digest = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    directory = Path(config.IMAGES_DIR) / digest[:20]
    directory.mkdir(parents=True, exist_ok=True)
    elements = []
    with fitz.open(pdf_path) as doc:
        if doc.needs_pass:
            raise ValueError('PDF requires a password; provide an unlocked local copy.')
        for index, page in enumerate(doc):
            page_meta = dict(page=index + 1, page_label=page.get_label() or str(index + 1))
            native_text = page.get_text('text', sort=True)
            # Dict extraction once per page, instead of once per text block.
            for block in page.get_text('dict', sort=True)['blocks']:
                if block.get('type') != 0:
                    continue
                lines = block.get('lines', [])
                text = '\n'.join(''.join(s['text'] for s in line['spans'])
                                 for line in lines).strip()
                if not text:
                    continue
                data = dict(text=text, spans=[s for line in lines for s in line['spans']])
                elements.append(dict(type='title' if _heading(
                    data) else 'body', text=text, **page_meta))
            # Explicit row/column serialization preserves tables better than loose blocks.
            try:
                for number, table in enumerate(page.find_tables().tables):
                    rows = table.extract()
                    text = '\n'.join(' | '.join((cell or '').replace('\n', ' ')
                                     for cell in row) for row in rows)
                    if text.strip():
                        elements.append(dict(type='table', text=text,
                                        table_number=number + 1, **page_meta))
            except Exception as exc:
                log.warning('Table extraction failed on PDF page %s: %s', index + 1, exc)
            scanned = len(native_text.strip()) < 30
            # Render the full page for scans and vector artwork: page.get_images()
            # alone cannot see vector diagrams or their spatial layout.
            rendered = scanned or (render_figures and bool(page.get_drawings()))
            if rendered:
                path = directory / f'page_{index + 1}.png'
                page.get_pixmap(dpi=config.OCR_DPI, alpha=False).save(str(path))
                elements.append(dict(type='image', text='', image_path=str(path),
                                     extraction='rendered_page_ocr', native_text=native_text, **page_meta))
            else:
                seen = set()
                for image in page.get_images(full=True):
                    xref = image[0]
                    if xref in seen:
                        continue
                    seen.add(xref)
                    try:
                        extracted = doc.extract_image(xref)
                        path = directory / f'page_{index + 1}_xref_{xref}.{extracted["ext"]}'
                        path.write_bytes(extracted['image'])
                        elements.append(dict(type='image', text='', image_path=str(path),
                                             extraction='embedded_image_ocr', **page_meta))
                    except Exception as exc:
                        log.warning('Image extraction failed on page %s: %s', index + 1, exc)
    return elements


def split_streams(elements):
    result = {key: [] for key in ('glossary', 'index', 'toc', 'body', 'images')}
    current = 'body'
    for element in elements:
        if element['type'] == 'image':
            # Do not retrieve illustrations from deliberately excluded back matter.
            if current not in ('index', 'toc'):
                result['images'].append(element)
            continue
        if element['type'] == 'title':
            title = element['text'].strip()
            if GLOSSARY_TITLE_RE.fullmatch(title):
                current = 'glossary'
            elif INDEX_TITLE_RE.fullmatch(title):
                current = 'index'
            elif TOC_TITLE_RE.fullmatch(title):
                current = 'toc'
            elif current != 'glossary' or len(title) != 1:
                # Glossary letter headings A/B/C retain glossary classification.
                current = 'body'
        result[current].append(element)
    return result


def parse_glossary_entries(elements):
    entries = []
    for el in elements:
        text = el['text'].strip()
        if not text or GLOSSARY_TITLE_RE.fullmatch(text):
            continue
        match = GLOSSARY_ENTRY_RE.match(text)
        entries.append(dict(term=match.group(1).strip() if match else '',
                            definition=match.group(2).strip() if match else text,
                            page_number=el['page'], page_label=el.get('page_label', str(el['page']))))
    return entries


def _make_docs(text, metadata):
    docs = []
    for start, end, part in split_with_offsets(text, config.CHUNK_MAX_CHARACTERS, config.CHUNK_OVERLAP):
        meta = dict(metadata, char_start=start, char_end=end)
        identity = json.dumps(meta, sort_keys=True, ensure_ascii=False) + '\n' + part
        docs.append(dict(id=hashlib.sha256(identity.encode()).hexdigest(), text=part, metadata=meta))
    return docs


def build_body_chunks(elements, source_doc):
    docs, parts = [], []
    page = None
    label = ''
    section = ''
    kind = 'text'

    def flush():
        if parts:
            docs.extend(_make_docs('\n\n'.join(parts), dict(source_doc=source_doc,
                                                            page_number=page, page_start=page, page_end=page,
                                                            page_label=label, type=kind, section=section)))
            parts.clear()

    for el in elements:
        text = el['text'].strip()
        if not text:
            continue
        new_kind = 'table' if el['type'] == 'table' else 'text'
        if el['page'] != page or el['type'] == 'title' or new_kind != kind:
            flush()
            page = el['page']
            label = el.get('page_label', str(page))
            kind = new_kind
        if el['type'] == 'title':
            section = text
        parts.append(text)
    flush()  # Keep final/trailing headings rather than silently losing them.
    return docs


def build_glossary_docs(entries, source_doc):
    docs = []
    for el in entries:
        text = f'{el["term"]}: {el["definition"]}' if el['term'] else el['definition']
        docs.extend(_make_docs(text, dict(source_doc=source_doc, page_number=el['page_number'],
                                          page_start=el['page_number'], page_end=el['page_number'],
                                          page_label=el.get('page_label', str(el['page_number'])), type='glossary', term=el['term'])))
    return docs


def ocr_image(image_path):
    command = config.TESSERACT_CMD or shutil.which('tesseract')
    if not command:
        raise RuntimeError('Tesseract not found. Add it to PATH or set PMBOK_TESSERACT_CMD.')
    result = subprocess.run([command, str(image_path), 'stdout', '-l', config.OCR_LANGUAGE],
                            check=True, capture_output=True, text=True,
                            encoding='utf-8', timeout=90)
    return result.stdout.strip()


def load_caption_cache():
    try:
        result = json.loads(Path(config.CAPTION_CACHE_FILE).read_text(encoding='utf-8'))
        return result if isinstance(result, dict) else {}
    except FileNotFoundError:
        return {}
    except (ValueError, OSError) as exc:
        log.warning('Ignoring unreadable OCR cache: %s', exc)
        return {}


def save_caption_cache(cache):
    path = Path(config.CAPTION_CACHE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='ocr_', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(cache, stream, ensure_ascii=False)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def process_images(elements, source_doc, ocr_fn=None):
    if not elements:
        return []
    cache, docs = load_caption_cache(), []
    ocr_fn = ocr_fn or ocr_image
    for element in elements:
        path = Path(element['image_path'])
        try:
            # Include OCR settings in the key; failures and blank outputs are not cached.
            key = hashlib.sha256(
                path.read_bytes() + f'ocr-v2:{config.OCR_LANGUAGE}:{config.TESSERACT_CMD}'.encode()).hexdigest()
            text = cache.get(key, {}).get('ocr_text')
            if text is None:
                text = ocr_fn(str(path))
                if text:
                    cache[key] = {'ocr_text': text}
            if not text:
                continue
            native_tokens = set(re.findall(r'\w+', element.get('native_text', '').lower()))
            ocr_tokens = set(re.findall(r'\w+', text.lower()))
            if native_tokens and ocr_tokens.issubset(native_tokens):
                continue  # Render added no searchable labels beyond native text.
            page = element['page']
            docs.extend(_make_docs(text, dict(source_doc=source_doc, page_number=page,
                                              page_start=page, page_end=page, page_label=element.get(
                                                  'page_label', str(page)),
                                              type='image', image_path=str(path), extraction=element.get('extraction', 'ocr'))))
        except Exception as exc:
            log.warning('OCR skipped for %s: %s', path.name, exc)
    save_caption_cache(cache)
    return docs


def ingest(pdf_path, doc_name, title=None, edition='', use_ocr=True, render_figures=True):
    elements = extract_pdf(pdf_path, render_figures=render_figures)
    streams = split_streams(elements)
    docs = build_body_chunks(streams['body'], doc_name)
    docs += build_glossary_docs(parse_glossary_entries(streams['glossary']), doc_name)
    if use_ocr:
        docs += process_images(streams['images'], doc_name)
    if not docs:
        raise ValueError('No usable text. For scanned PDFs, enable OCR and install Tesseract.')
    file_hash = hashlib.sha256(Path(pdf_path).read_bytes()).hexdigest()
    # Stable occurrence IDs preserve repeated identical passages on the same page.
    occurrences = {}
    for doc in docs:
        base = doc['id']
        occurrence = occurrences.get(base, 0)
        occurrences[base] = occurrence + 1
        doc['id'] = f'{base}_{occurrence}'
        doc['metadata'].update(document_title=title or doc_name, edition=edition,
                               file_sha256=file_hash, chunking_version=config.CHUNKING_VERSION)
    from service import load_embedding_model
    name = store_documents(docs, load_embedding_model(), doc_name)
    print(f'Indexed {len(docs)} chunks as {name}. Reload open query sessions to use this snapshot.')
    return docs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pdf_path')
    parser.add_argument('--doc-name')
    parser.add_argument('--title')
    parser.add_argument('--edition', default='')
    parser.add_argument('--no-ocr', action='store_true')
    parser.add_argument('--no-render-figures', action='store_true')
    args = parser.parse_args()
    if not Path(args.pdf_path).is_file():
        parser.error(f'PDF not found: {args.pdf_path}')
    logging.basicConfig(level=logging.INFO)
    start = time.monotonic()
    ingest(args.pdf_path, args.doc_name or Path(args.pdf_path).stem, args.title,
           args.edition, not args.no_ocr, not args.no_render_figures)
    print(f'Done in {time.monotonic() - start:.1f}s.')


if __name__ == '__main__':
    main()
