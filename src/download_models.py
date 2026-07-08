"""Download/cache the embedding model and reranker before enabling offline mode."""
import os

import config
from service import load_embedding_model, load_reranker

if __name__ == '__main__':
    if os.environ.get('HF_HUB_OFFLINE') == '1':
        raise SystemExit('Disable PMBOK_OFFLINE and HF_HUB_OFFLINE before downloading models.')
    load_embedding_model()
    load_reranker()
    print('Embedding model and reranker cached. Also run: ollama pull ' + config.LLM_MODEL)
