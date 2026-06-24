"""Set PMBOK_* environment variables before starting the application."""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = str(Path(os.getenv('PMBOK_CHROMA_DIR', PROJECT_DIR / 'chroma_db')))
IMAGES_DIR = str(Path(os.getenv('PMBOK_IMAGES_DIR', PROJECT_DIR / 'images')))
CAPTION_CACHE_FILE = str(PROJECT_DIR / 'data' / 'caption_cache.json')
COLLECTION_NAME = 'pdf_knowledge_base'
INDEX_STATE_FILE = str(Path(CHROMA_DIR) / 'active_index.json')
INDEX_SCHEMA_VERSION = 2
CHUNKING_VERSION = 'page-section-offset-v2'
EMBEDDING_MODEL_NAME = os.getenv('PMBOK_EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5')
EMBEDDING_MODEL_REVISION = os.getenv('PMBOK_EMBEDDING_REVISION', '')
CROSS_ENCODER_MODEL = os.getenv('PMBOK_RERANKER', 'cross-encoder/ms-marco-MiniLM-L-4-v2')
CROSS_ENCODER_REVISION = os.getenv('PMBOK_RERANKER_REVISION', '')
OLLAMA_HOST = os.getenv('PMBOK_OLLAMA_HOST', 'http://localhost:11434').rstrip('/')
LLM_MODEL = os.getenv('PMBOK_LLM_MODEL', 'qwen2.5:3b')
CHUNK_MAX_CHARACTERS = int(os.getenv('PMBOK_CHUNK_SIZE', '800'))
CHUNK_OVERLAP = int(os.getenv('PMBOK_CHUNK_OVERLAP', '100'))
TOP_K = int(os.getenv('PMBOK_TOP_K', '3'))
CANDIDATE_K = int(os.getenv('PMBOK_CANDIDATE_K', '10'))
CONTEXT_TOKEN_BUDGET = int(os.getenv('PMBOK_CONTEXT_TOKENS', '3000'))
LLM_CONTEXT_TOKENS = int(os.getenv('PMBOK_LLM_CONTEXT', '8192'))
MAX_QUESTION_CHARACTERS = 4000
CHROMA_DISTANCE_METRIC = 'cosine'
OCR_LANGUAGE = os.getenv('PMBOK_OCR_LANGUAGE', 'eng')
TESSERACT_CMD = os.getenv('PMBOK_TESSERACT_CMD', '')
OCR_DPI = 150
# Raw MS-MARCO scores are not probabilities. Calibrate on labelled questions.
MIN_RERANK_SCORE = float(os.getenv('PMBOK_MIN_RERANK_SCORE', '0.0'))
if os.getenv('PMBOK_OFFLINE', '0') == '1':
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'


def index_signature():
    return dict(schema_version=INDEX_SCHEMA_VERSION,
                embedding_model=EMBEDDING_MODEL_NAME,
                embedding_revision=EMBEDDING_MODEL_REVISION,
                chunking_version=CHUNKING_VERSION,
                chunk_size=CHUNK_MAX_CHARACTERS, chunk_overlap=CHUNK_OVERLAP,
                distance_metric=CHROMA_DISTANCE_METRIC)
