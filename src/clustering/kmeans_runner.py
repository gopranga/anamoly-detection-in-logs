import json
import logging
import os
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans

from .k_selection import evaluate_k_range, save_k_evaluations, suggest_k

LOGGER = logging.getLogger(__name__)


@dataclass
class KMeansConfig:
    method: str = 'minibatch'  # 'kmeans' | 'minibatch'
    batch_size: int = 2048
    sample_size: int = 10000
    max_iter: int = 100
    n_init: int = 10
    random_state: int = 0
    k_values_start: int = 2
    k_values_stop: int = 64
    k_values_step: int = 2


def load_config(path: str) -> Dict:
    import yaml
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def ensure_results_dirs(base_dir: str) -> Dict[str, str]:
    cluster_dir = os.path.join(base_dir, 'clustering')
    os.makedirs(cluster_dir, exist_ok=True)
    return {
        'cluster_dir': cluster_dir,
        'eval_json': os.path.join(cluster_dir, 'k_evaluations.json'),
        'labels_npy': os.path.join(cluster_dir, 'labels.npy'),
        'model_json': os.path.join(cluster_dir, 'model_summary.json'),
        'centers_npy': os.path.join(cluster_dir, 'cluster_centers.npy'),
    }


def run_kmeans_pipeline(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    cfg = load_config(config_path)
    out_dir = cfg['paths']['output_dir']
    embeddings_path = os.path.join(out_dir, 'regex_embeddings.npy')
    paths = ensure_results_dirs(out_dir)

    # Hyperparams with safe defaults; allow overrides via cfg['clustering']
    cluster_cfg_dict = (cfg.get('clustering') or {}).copy()
    cluster_cfg = KMeansConfig(**{**asdict(KMeansConfig()), **cluster_cfg_dict})

    # Construct K range
    # Limit max K by number of points
    X = np.load(embeddings_path, mmap_mode='r')  # type: ignore
    n = X.shape[0]
    k_values = [
        k for k in range(cluster_cfg.k_values_start, min(cluster_cfg.k_values_stop, max(2, n)) + 1, max(1, cluster_cfg.k_values_step))
        if k <= n
    ]
    if not k_values:
        LOGGER.error("No valid K range computed. n=%d", n)
        return

    LOGGER.info("Evaluating K in %s using method=%s ...", k_values, cluster_cfg.method)
    evals = evaluate_k_range(
        embeddings_path=embeddings_path,
        k_values=k_values,
        method=cluster_cfg.method,
        batch_size=cluster_cfg.batch_size,
        sample_size=cluster_cfg.sample_size,
        max_iter=cluster_cfg.max_iter,
        n_init=cluster_cfg.n_init,
        random_state=cluster_cfg.random_state,
    )
    suggestion = suggest_k(evals)
    save_k_evaluations(paths['eval_json'], evals, suggestion)
    LOGGER.info("Saved K evaluations and suggestions to %s", paths['eval_json'])

    best_k = suggestion.get('consensus_k') or suggestion.get('silhouette_best') or suggestion.get('elbow_best')
    if not best_k:
        # Fallback to a heuristic: sqrt(n/2)
        best_k = max(2, int((n / 2) ** 0.5))
        LOGGER.warning("No consensus K; falling back to heuristic K=%d", best_k)

    LOGGER.info("Training final model with K=%d using %s", best_k, cluster_cfg.method)
    if cluster_cfg.method == 'kmeans':
        final_model = KMeans(
            n_clusters=int(best_k),
            n_init=cluster_cfg.n_init,
            max_iter=cluster_cfg.max_iter,
            random_state=cluster_cfg.random_state,
            tol=1e-4,
            algorithm='lloyd',
        )
        final_model.fit(np.asarray(X))
        labels = final_model.labels_
        centers = final_model.cluster_centers_
        inertia_val = float(final_model.inertia_)
    else:
        final_model = MiniBatchKMeans(
            n_clusters=int(best_k),
            batch_size=cluster_cfg.batch_size,
            max_iter=cluster_cfg.max_iter,
            n_init=cluster_cfg.n_init,
            random_state=cluster_cfg.random_state,
            reassignment_ratio=0.01,
        )
        # Two passes: fit then predict
        for i in range(0, n, final_model.batch_size):
            j = min(n, i + final_model.batch_size)
            final_model.partial_fit(np.asarray(X[i:j]))
        labels = np.empty(n, dtype=int)
        for i in range(0, n, final_model.batch_size):
            j = min(n, i + final_model.batch_size)
            labels[i:j] = final_model.predict(np.asarray(X[i:j]))
        centers = final_model.cluster_centers_
        # Compute inertia accurately against centers
        inertia_val = 0.0
        for i in range(0, n, final_model.batch_size):
            j = min(n, i + final_model.batch_size)
            chunk = np.asarray(X[i:j])
            d2 = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            inertia_val += float(np.min(d2, axis=1).sum())

    # Persist artifacts
    np.save(paths['labels_npy'], labels)
    np.save(paths['centers_npy'], centers)
    with open(paths['model_json'], 'w', encoding='utf-8') as fh:
        json.dump({
            'method': cluster_cfg.method,
            'k': int(best_k),
            'inertia': inertia_val,
            'n_samples': int(n),
            'n_features': int(X.shape[1]),
            'params': asdict(cluster_cfg),
        }, fh, indent=2)

    LOGGER.info("Saved clustering artifacts: %s, %s, %s", paths['labels_npy'], paths['centers_npy'], paths['model_json'])


