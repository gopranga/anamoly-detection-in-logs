# src/clustering/k_selection.py
"""
Provides a robust and comprehensive suite of functions to evaluate a range of k
values for KMeans-based clustering. This module is based on the user's original,
advanced implementation, which includes multiple evaluation metrics, consensus-based
suggestions, and memory-safe streaming for large datasets.
"""

import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


@dataclass
class KEvaluation:
    """Dataclass to hold the evaluation results for a single k value."""

    k: int
    inertia: Optional[float]
    silhouette: Optional[float]
    davies_bouldin: Optional[float]
    calinski_harabasz: Optional[float]
    train_seconds: float
    sample_size: int


def _iter_chunks_memmap(
        data: np.memmap, batch_size: int
) -> Generator[np.ndarray, None, None]:
    """Iterates over a memory-mapped numpy array in chunks."""
    num_samples = data.shape[0]
    for i in range(0, num_samples, batch_size):
        yield np.asarray(data[i: min(num_samples, i + batch_size)])


def _predict_streaming(model: Any, data: np.memmap, batch_size: int) -> np.ndarray:
    """Predicts labels for a large dataset in a streaming fashion."""
    labels = np.empty(data.shape[0], dtype=int)
    position = 0
    for chunk in _iter_chunks_memmap(data, batch_size):
        chunk_labels = model.predict(chunk)
        labels[position: position + len(chunk_labels)] = chunk_labels
        position += len(chunk_labels)
    return labels


def _compute_inertia_streaming(
        centers: np.ndarray, data: np.memmap, batch_size: int
) -> float:
    """Accurately computes inertia for a large dataset in a streaming fashion."""
    inertia = 0.0
    for chunk in _iter_chunks_memmap(data, batch_size):
        # Shape: (batch, centers)
        distances_sq = ((chunk[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        inertia += np.min(distances_sq, axis=1).sum()
    return float(inertia)


def _fit_model_for_k(
        data: np.memmap, k: int, params: Dict[str, Any]
) -> Tuple[Any, float]:
    """Fits a KMeans or MiniBatchKMeans model for a given k."""
    start_time = time.time()
    method = params["method"]

    if method == "kmeans":
        model = KMeans(
            n_clusters=k,
            n_init=params["n_init"],
            max_iter=params["max_iter"],
            random_state=params["random_state"],
            algorithm="lloyd",
        )
        model.fit(np.asarray(data))
    else:  # minibatch
        model = MiniBatchKMeans(
            n_clusters=k,
            batch_size=params["batch_size"],
            max_iter=params["max_iter"],
            n_init=params["n_init"],
            random_state=params["random_state"],
        )
        for chunk in _iter_chunks_memmap(data, params["batch_size"]):
            model.partial_fit(chunk)

    elapsed_time = time.time() - start_time
    return model, elapsed_time


def _safe_metric(name: str, func: Callable[[], float]) -> Optional[float]:
    """Safely computes a metric, returning None on failure."""
    try:
        value = float(func())
        return value if math.isfinite(value) else None
    except Exception as e:
        logging.debug(f"Metric calculation for '{name}' failed: {e}")
        return None


def evaluate_k_range(
        embeddings_path: str, k_values: Sequence[int], params: Dict[str, Any]
) -> List[KEvaluation]:
    """
    Evaluates a range of K values with multiple metrics.

    Args:
        embeddings_path: Path to the memory-mapped embeddings file.
        k_values: A sequence of integers for k to evaluate.
        params: A dictionary of clustering parameters.

    Returns:
        A list of KEvaluation objects with the results for each k.
    """
    data = np.load(embeddings_path, mmap_mode="r")
    num_samples = data.shape[0]
    rng = np.random.RandomState(params["random_state"])
    sample_indices = rng.choice(
        num_samples, size=min(params["sample_size"], num_samples), replace=False
    )
    data_sample = np.asarray(data[sample_indices])

    results: List[KEvaluation] = []
    for k in k_values:
        model, train_sec = _fit_model_for_k(data, k, params)
        if params["method"] == "kmeans":
            labels = model.labels_
            inertia = float(model.inertia_)
        else:  # minibatch
            labels = _predict_streaming(model, data, params["batch_size"])
            inertia = _compute_inertia_streaming(
                np.asarray(model.cluster_centers_), data, params["batch_size"]
            )

        labels_sample = labels[sample_indices]
        sil = _safe_metric(
            "silhouette", lambda: silhouette_score(data_sample, labels_sample)
        )
        db = _safe_metric(
            "davies_bouldin",
            lambda: davies_bouldin_score(data_sample, labels_sample),
        )
        ch = _safe_metric(
            "calinski_harabasz",
            lambda: calinski_harabasz_score(data_sample, labels_sample),
        )

        results.append(
            KEvaluation(
                k=k,
                inertia=inertia,
                silhouette=sil,
                davies_bouldin=db,
                calinski_harabasz=ch,
                train_seconds=train_sec,
                sample_size=len(sample_indices),
            )
        )
    return results


def _elbow_k(k_list: List[int], inertia_list: List[float]) -> Optional[int]:
    """Finds the elbow point in an inertia curve."""
    if len(k_list) < 3:
        return None
    p1 = np.array([k_list[0], inertia_list[0]])
    p2 = np.array([k_list[-1], inertia_list[-1]])

    distances = []
    for i in range(len(k_list)):
        p3 = np.array([k_list[i], inertia_list[i]])
        distance = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        distances.append(distance)

    return k_list[np.argmax(distances)] if distances else None


def suggest_k(evals: List[KEvaluation]) -> Dict[str, Optional[int]]:
    """Suggests the best k based on a consensus of multiple metrics."""
    if not evals:
        return {}

    k_by_sil = max(
        [(e.k, e.silhouette) for e in evals if e.silhouette is not None],
        key=lambda x: x[1],
        default=(None, None),
    )[0]
    k_by_db = min(
        [(e.k, e.davies_bouldin) for e in evals if e.davies_bouldin is not None],
        key=lambda x: x[1],
        default=(None, None),
    )[0]
    k_by_ch = max(
        [
            (e.k, e.calinski_harabasz)
            for e in evals
            if e.calinski_harabasz is not None
        ],
        key=lambda x: x[1],
        default=(None, None),
    )[0]

    inertia_k = [e.k for e in evals if e.inertia is not None]
    inertia_vals = [e.inertia for e in evals if e.inertia is not None]
    k_by_elbow = _elbow_k(inertia_k, inertia_vals)

    votes = [k for k in [k_by_sil, k_by_db, k_by_ch, k_by_elbow] if k is not None]
    consensus_k = int(np.bincount(votes).argmax()) if votes else None

    return {
        "silhouette_best": k_by_sil,
        "davies_bouldin_best": k_by_db,
        "calinski_harabasz_best": k_by_ch,
        "elbow_best": k_by_elbow,
        "consensus_k": consensus_k,
    }


def save_k_evaluations(
        path: str, evaluations: List[KEvaluation], suggestion: Dict[str, Optional[int]]
) -> None:
    """Saves the k evaluation results and suggestions to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "evaluations": [asdict(e) for e in evaluations],
        "suggestion": suggestion,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
