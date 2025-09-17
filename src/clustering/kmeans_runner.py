# src/clustering/kmeans_runner.py
"""
Main script for the training workflow. It orchestrates the process of
evaluating the optimal k, training a final KMeans or MiniBatchKMeans model,
and saving all necessary artifacts for the anomaly detection phase.
This script is based on the user's original, robust implementation.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans, MiniBatchKMeans

from .k_selection import (
    _compute_inertia_streaming,
    _iter_chunks_memmap,
    _predict_streaming,
    evaluate_k_range,
    save_k_evaluations,
    suggest_k,
)


def run_kmeans_pipeline(config: Dict[str, Any]) -> None:
    """
    Executes the full KMeans training and evaluation pipeline.

    Args:
        config: The project configuration dictionary.
    """
    paths_cfg = config["paths"]
    cluster_cfg = config.get("clustering", {})
    k_selection_cfg = cluster_cfg.get("k_selection", {})

    output_dir = paths_cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # --- Prepare output paths ---
    embeddings_path = os.path.join(output_dir, paths_cfg["output_embeddings_file"])
    eval_json_path = os.path.join(output_dir, paths_cfg["k_evaluations_file"])
    labels_npy_path = os.path.join(output_dir, paths_cfg["labels_file"])
    model_json_path = os.path.join(output_dir, paths_cfg["model_summary_file"])
    centers_npy_path = os.path.join(output_dir, paths_cfg["cluster_centers_file"])

    # --- Load embeddings and determine K range ---
    if not os.path.exists(embeddings_path):
        logging.error(f"Embeddings file not found at {embeddings_path}. Halting.")
        return

    data = np.load(embeddings_path, mmap_mode="r")
    num_samples = data.shape[0]

    # --- FIX: Determine Best K based on config ---
    k_value = cluster_cfg.get("k_value", "auto")
    best_k: int

    if isinstance(k_value, int) and k_value > 1:
        logging.info(f"Using fixed number of clusters from config: k = {k_value}")
        best_k = k_value
    else:
        logging.info("Automatic k-selection enabled ('k_value: auto').")
        k_start = k_selection_cfg.get("start", 2)
        k_stop = min(k_selection_cfg.get("stop", 50), num_samples)
        k_step = k_selection_cfg.get("step", 2)
        k_values = range(k_start, k_stop + 1, k_step)

        if not k_values:
            logging.error(f"No valid K range to evaluate for {num_samples} samples.")
            return

        logging.info(f"Evaluating K in range {list(k_values)}...")
        eval_params = {**cluster_cfg, **k_selection_cfg}
        evaluations = evaluate_k_range(embeddings_path, k_values, eval_params)
        suggestion = suggest_k(evaluations)
        save_k_evaluations(eval_json_path, evaluations, suggestion)

        best_k = (
                suggestion.get("consensus_k")
                or suggestion.get("silhouette_best")
                or int((num_samples / 2) ** 0.5)
        )
        logging.info(f"Saved K evaluations and suggestions to {eval_json_path}")

    # --- Train Final Model ---
    logging.info(f"Training final model with K={best_k} using '{cluster_cfg['method']}'...")

    # --- FIX: Initialize variables before the conditional block ---
    final_model: Optional[Any] = None
    labels: Optional[np.ndarray] = None
    centers: Optional[np.ndarray] = None
    inertia: Optional[float] = None

    if cluster_cfg["method"] == "kmeans":
        final_model = KMeans(
            n_clusters=best_k,
            n_init=cluster_cfg.get("n_init", 10),
            max_iter=cluster_cfg.get("max_iter", 300),
            random_state=cluster_cfg.get("random_state", 42),
        )
        final_model.fit(data)
        labels = final_model.labels_
        centers = final_model.cluster_centers_
        inertia = float(final_model.inertia_)
    else:  # minibatch
        final_model = MiniBatchKMeans(
            n_clusters=best_k,
            batch_size=cluster_cfg.get("batch_size", 2048),
            max_iter=cluster_cfg.get("max_iter", 100),
            n_init=cluster_cfg.get("n_init", "auto"),
            random_state=cluster_cfg.get("random_state", 42),
        )
        for chunk in _iter_chunks_memmap(data, final_model.batch_size):
            final_model.partial_fit(chunk)
        labels = _predict_streaming(final_model, data, final_model.batch_size)
        centers = final_model.cluster_centers_
        inertia = _compute_inertia_streaming(centers, data, final_model.batch_size)

    """
    # ******** Solution::1 Common distance threshold for all clusters *********#
    
    # --- FIX: Calculate and save the distance threshold ---
    logging.info("Calculating anomaly distance threshold from training data...")
    distances = cdist(data, centers).min(axis=1)
    percentile_threshold = cluster_cfg.get("distance_threshold", 95)
    distance_threshold = np.percentile(distances, percentile_threshold)
    logging.info(f"Distance threshold at {percentile_threshold}th percentile: {distance_threshold:.4f}")

    # --- Persist Final Artifacts ---
    if labels is None or centers is None or inertia is None:
        logging.error("Model training failed, artifacts will not be saved.")
        return

    np.save(labels_npy_path, labels)
    np.save(centers_npy_path, centers)
    with open(model_json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "method": cluster_cfg["method"],
                "k": best_k,
                "inertia": inertia,
                "n_samples": num_samples,
                "n_features": data.shape[1],
                "params": cluster_cfg,
                # --- FIX: Add the threshold and centers path to the summary ---
                "distance_threshold": distance_threshold,
                "cluster_centers_file": os.path.abspath(centers_npy_path),
            },
            fh,
            indent=2,
        )
    logging.info(
        f"Saved final clustering artifacts to directory: {os.path.dirname(model_json_path)}"
    )
    """

    # ******** Solution::2 Distance threshold for per cluster basis *********#
    # --- FIX: Calculate and save PER-CLUSTER thresholds ---
    logging.info("Calculating per-cluster anomaly distance thresholds...")

    # First, get the labels for the training data
    labels = _predict_streaming(final_model, data, cluster_cfg.get("batch_size", 2048))

    per_cluster_thresholds: Dict[int, float] = {}
    percentile_threshold = cluster_cfg.get("distance_threshold", 95)

    for i in range(best_k):
        # Get all the embeddings that belong to this cluster
        cluster_mask = (labels == i)
        if np.any(cluster_mask):
            cluster_embeddings = data[cluster_mask]

            # Calculate distances for ONLY the points in this cluster
            distances_in_cluster = cdist(cluster_embeddings, [centers[i]]).flatten()

            # Calculate the threshold for THIS cluster
            per_cluster_thresholds[i] = np.percentile(distances_in_cluster, percentile_threshold)

    logging.info(f"Calculated thresholds for {len(per_cluster_thresholds)} clusters.")

    # --- Persist Final Artifacts ---
    np.save(labels_npy_path, labels)
    np.save(centers_npy_path, centers)
    with open(model_json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "method": cluster_cfg["method"],
                "k": best_k,
                "inertia": inertia,
                # --- FIX: Save the DICTIONARY of thresholds ---
                "per_cluster_thresholds": per_cluster_thresholds,
                "cluster_centers_file": os.path.abspath(centers_npy_path),
            },
            fh,
            indent=2,
        )
    logging.info(
        f"Saved final clustering artifacts, including model summary to {model_json_path}"
    )
