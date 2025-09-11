import logging
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

LOGGER = logging.getLogger(__name__)


def _embed_sentence_transformers(texts: List[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return model.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True)


def _embed_tfidf_svd(texts: List[str], dims: int = 64) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X = tfidf.fit_transform(texts)
    k = min(dims, max(2, X.shape[1] - 1))
    svd = TruncatedSVD(n_components=k, random_state=0)
    V = svd.fit_transform(X)
    # L2 normalize
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    return V


def _embed_hash(texts: List[str], dims: int = 64) -> np.ndarray:
    import re as _re
    M = len(texts)
    mat = np.zeros((M, dims), dtype=float)
    for i, t in enumerate(texts):
        for tok in _re.findall(r"\w+|<\*>|[^\w\s]", t.lower()):
            mat[i, hash(tok) % dims] += 1.0
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    return mat


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def main(config_path: str = 'src/configs/project.yaml') -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    cfg = load_config(config_path)
    out_dir = cfg['paths']['output_dir']
    corrected_jsonl = os.path.join(out_dir, cfg['paths']['corrected_templates_jsonl'])
    emb_out_jsonl = os.path.join(out_dir, cfg['paths']['embeddings_jsonl'])
    emb_out_npy = os.path.join(out_dir, 'regex_embeddings.npy')  # Path for NumPy file export
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(corrected_jsonl):
        LOGGER.error("Corrected templates JSONL not found at %s. Run the Drain3 pipeline first.", corrected_jsonl)
        return

    df = pd.read_json(corrected_jsonl, orient='records', lines=True)
    texts = df['corrected_template'].astype(str).tolist()
    if not texts:
        LOGGER.warning("No templates to embed.")
        return

    try:
        V = _embed_sentence_transformers(texts)
        method = 'sentence-transformers/all-MiniLM-L6-v2'
    except Exception as e1:
        LOGGER.info("sentence-transformers unavailable (%s). Falling back to TF-IDF+SVD.", e1)
        try:
            V = _embed_tfidf_svd(texts)
            method = 'tfidf+svd'
        except Exception as e2:
            LOGGER.info("TF-IDF+SVD failed (%s). Falling back to simple hashing.", e2)
            V = _embed_hash(texts)
            method = 'simple-hash'

    # Normalize the embeddings for safety
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    # Export the embeddings as JSONL
    emb_df = pd.DataFrame(V, columns=[f'e{i}' for i in range(V.shape[1])])
    out_df = pd.concat([df[['cluster_id', 'corrected_template']].reset_index(drop=True), emb_df], axis=1)
    out_df.to_json(emb_out_jsonl, orient='records', lines=True)

    # Export the embeddings as a NumPy file
    np.save(emb_out_npy, V)

    LOGGER.info("Saved %d embeddings to %s (JSONL) and %s (NumPy) using %s",
                len(out_df), emb_out_jsonl, emb_out_npy, method)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='../configs/project.yaml')
    args = ap.parse_args()
    main(args.config)
