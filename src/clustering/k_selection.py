import json
import logging
import math
import os
import time
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

LOGGER = logging.getLogger(__name__)


def _uniform_sample_indices(num_items: int, sample_size: int, rng: np.random.RandomState) -> np.ndarray:
    sample_size = int(max(1, min(sample_size, num_items)))
    if sample_size >= num_items:
        return np.arange(num_items)
    return rng.choice(num_items, size=sample_size, replace=False)


def _iter_chunks_memmap(X: np.memmap, batch_size: int) -> Iterable[Tuple[int, int, np.ndarray]]:
    n = X.shape[0]
    if batch_size <= 0:
        batch_size = n
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        yield i, j, np.asarray(X[i:j])


def _predict_streaming(model, X: np.memmap, batch_size: int) -> np.ndarray:
    labels = np.empty(X.shape[0], dtype=int)
    pos = 0
    for i, j, chunk in _iter_chunks_memmap(X, batch_size):
        labels[i:j] = model.predict(chunk)
        pos = j
    return labels[:pos]


def _compute_inertia_streaming(centers: np.ndarray, X: np.memmap, batch_size: int) -> float:
    # sum of squared distances to nearest center
    inertia = 0.0
    for _, _, chunk in _iter_chunks_memmap(X, batch_size):
        # (batch, centers)
        d2 = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        inertia += np.min(d2, axis=1).sum()
    return float(inertia)


def _fit_model_for_k(
    X: np.memmap,
    k: int,
    method: str,
    batch_size: int,
    max_iter: int,
    n_init: int,
    random_state: int,
) -> Tuple[object, float]:
    t0 = time.time()
    if method == 'kmeans':
        model = KMeans(
            n_clusters=k,
            n_init=n_init,
            max_iter=max_iter,
            random_state=random_state,
            tol=1e-4,
            algorithm='lloyd',
        )
        # KMeans needs full array; memmaps are fine
        model.fit(np.asarray(X))
        elapsed = time.time() - t0
        return model, elapsed
    else:
        # MiniBatchKMeans with streaming partial_fit
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=batch_size if batch_size > 0 else 1024,
            max_iter=max_iter,
            n_init=n_init,
            random_state=random_state,
            reassignment_ratio=0.01,
        )
        # Use the sklearn 1.4+ fit interface directly; it accepts memmap, but to
        # respect memory constraints we call partial_fit in chunks
        for _, _, chunk in _iter_chunks_memmap(X, model.batch_size):
            model.partial_fit(chunk)
        elapsed = time.time() - t0
        return model, elapsed


@dataclass
class KEvaluation:
    k: int
    inertia: Optional[float]
    silhouette: Optional[float]
    davies_bouldin: Optional[float]
    calinski_harabasz: Optional[float]
    train_seconds: float
    sample_size: int


def _safe_metric(name: str, func: Callable[[], float]) -> Optional[float]:
    try:
        val = float(func())
        if math.isfinite(val):
            return val
        return None
    except Exception as e:
        LOGGER.debug("Metric %s failed: %s", name, e)
        return None


def evaluate_k_range(
    embeddings_path: str,
    k_values: Sequence[int],
    method: str = 'minibatch',
    batch_size: int = 2048,
    sample_size: int = 10000,
    max_iter: int = 100,
    n_init: int = 10,
    random_state: int = 0,
) -> List[KEvaluation]:
    """
    Evaluate a range of K values with multiple metrics.

    - Fits model (KMeans or MiniBatchKMeans)
    - Predicts labels for entire dataset (streaming for MiniBatch)
    - Computes: inertia (exact, streaming for MiniBatch), silhouette, DB, CH on a sample
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    X = np.load(embeddings_path, mmap_mode='r')  # type: ignore
    assert X.ndim == 2, "Embeddings must be 2D array"

    n_samples = X.shape[0]
    rng = np.random.RandomState(random_state)
    sample_idx = _uniform_sample_indices(n_samples, sample_size, rng)
    Xs = np.asarray(X[sample_idx])

    results: List[KEvaluation] = []
    for k in k_values:
        if k < 2:
            continue
        if k > n_samples:
            LOGGER.info("Skipping K=%d > num_samples=%d", k, n_samples)
            continue
        model, train_sec = _fit_model_for_k(X, k, method, batch_size, max_iter, n_init, random_state)
        if method == 'kmeans':
            labels = model.labels_  # type: ignore
            inertia_val = float(model.inertia_)  # type: ignore
        else:
            labels = _predict_streaming(model, X, batch_size)
            # recompute inertia against learned centers for accuracy
            inertia_val = _compute_inertia_streaming(np.asarray(model.cluster_centers_), X, batch_size)  # type: ignore

        # Sampled metrics for cost/perf balance
        # Map sample indices through labels
        labels_s = labels[sample_idx]

        sil = _safe_metric('silhouette', lambda: silhouette_score(Xs, labels_s, metric='euclidean'))
        db = _safe_metric('davies_bouldin', lambda: davies_bouldin_score(Xs, labels_s))
        ch = _safe_metric('calinski_harabasz', lambda: calinski_harabasz_score(Xs, labels_s))

        results.append(KEvaluation(
            k=k,
            inertia=inertia_val,
            silhouette=sil,
            davies_bouldin=db,
            calinski_harabasz=ch,
            train_seconds=train_sec,
            sample_size=int(len(sample_idx)),
        ))

    return results


def _elbow_k(k_list: List[int], inertia_list: List[float]) -> Optional[int]:
    if len(k_list) < 3:
        return None
    k_min, k_max = k_list[0], k_list[-1]
    i_min, i_max = inertia_list[0], inertia_list[-1]
    # Line through endpoints: distance of each point to line
    dk = k_max - k_min
    di = i_max - i_min
    denom = math.hypot(dk, di)
    if denom == 0:
        return None
    max_dist = -1.0
    best_k = None
    for k, i in zip(k_list, inertia_list):
        # Compute perpendicular distance to the line through endpoints
        # Using formula for point-line distance in 2D
        num = abs(di * (k - k_min) - dk * (i - i_min))
        d = num / denom
        if d > max_dist:
            max_dist = d
            best_k = k
    return best_k


def suggest_k(evals: List[KEvaluation]) -> Dict[str, Optional[int]]:
    if not evals:
        return {
            'silhouette_best': None,
            'davies_bouldin_best': None,
            'calinski_harabasz_best': None,
            'elbow_best': None,
            'consensus_k': None,
        }

    # Filter out None metrics
    k_vals = [e.k for e in evals]
    inertia_vals = [e.inertia for e in evals if e.inertia is not None]
    inertia_k = [e.k for e in evals if e.inertia is not None]

    silhouette_best = None
    db_best = None
    ch_best = None

    # Choose K by maximizing/minimizing metrics as appropriate
    sil_pairs = [(e.k, e.silhouette) for e in evals if e.silhouette is not None]
    if sil_pairs:
        silhouette_best = max(sil_pairs, key=lambda x: x[1])[0]
    db_pairs = [(e.k, e.davies_bouldin) for e in evals if e.davies_bouldin is not None]
    if db_pairs:
        db_best = min(db_pairs, key=lambda x: x[1])[0]
    ch_pairs = [(e.k, e.calinski_harabasz) for e in evals if e.calinski_harabasz is not None]
    if ch_pairs:
        ch_best = max(ch_pairs, key=lambda x: x[1])[0]

    elbow_best = _elbow_k(inertia_k, inertia_vals) if inertia_vals else None

    # Consensus: majority vote across available suggestions
    votes: List[int] = [k for k in [silhouette_best, db_best, ch_best, elbow_best] if k is not None]
    consensus = None
    if votes:
        # Count occurrences
        vals, counts = np.unique(np.array(votes), return_counts=True)
        consensus = int(vals[int(np.argmax(counts))])

    return {
        'silhouette_best': silhouette_best,
        'davies_bouldin_best': db_best,
        'calinski_harabasz_best': ch_best,
        'elbow_best': elbow_best,
        'consensus_k': consensus,
    }


def save_k_evaluations(path: str, evaluations: List[KEvaluation], suggestion: Dict[str, Optional[int]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'evaluations': [asdict(e) for e in evaluations],
        'suggestion': suggestion,
    }
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2)


