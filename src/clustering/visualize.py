import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


LOGGER = logging.getLogger(__name__)


def _safe_mpl_import():
    import matplotlib
    # Non-interactive backend for servers/CI
    try:
        matplotlib.use('Agg')
    except Exception:
        pass
    import matplotlib.pyplot as plt
    return plt


def _ensure_plots_dir(base_results_dir: str) -> str:
    plots_dir = os.path.join(base_results_dir, 'clustering', 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def _load_project_config(path: str) -> Dict:
    import yaml
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def plot_k_evaluations(results_dir: str) -> Optional[str]:
    plt = _safe_mpl_import()
    eval_path = os.path.join(results_dir, 'clustering', 'k_evaluations.json')
    if not os.path.exists(eval_path):
        LOGGER.warning("k_evaluations.json not found at %s", eval_path)
        return None

    with open(eval_path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    evaluations = data.get('evaluations') or []
    suggestion = data.get('suggestion') or {}

    if not evaluations:
        LOGGER.warning("No evaluations in %s", eval_path)
        return None

    k_vals = [int(e['k']) for e in evaluations]
    inertia_vals = [e.get('inertia') for e in evaluations]
    sil_vals = [e.get('silhouette') for e in evaluations]
    db_vals = [e.get('davies_bouldin') for e in evaluations]
    ch_vals = [e.get('calinski_harabasz') for e in evaluations]
    t_vals = [e.get('train_seconds') for e in evaluations]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(k_vals, inertia_vals, marker='o')
    ax.set_title('Inertia vs K (Elbow)')
    ax.set_xlabel('K')
    ax.set_ylabel('Inertia (lower better)')

    ax = axes[0, 1]
    ax.plot(k_vals, sil_vals, marker='o', color='green')
    ax.set_title('Silhouette vs K')
    ax.set_xlabel('K')
    ax.set_ylabel('Silhouette (higher better)')

    ax = axes[1, 0]
    ax.plot(k_vals, db_vals, marker='o', color='orange')
    ax.set_title('Davies-Bouldin vs K')
    ax.set_xlabel('K')
    ax.set_ylabel('DB (lower better)')

    ax = axes[1, 1]
    ax.plot(k_vals, ch_vals, marker='o', color='purple')
    ax.set_title('Calinski-Harabasz vs K')
    ax.set_xlabel('K')
    ax.set_ylabel('CH (higher better)')

    # Annotate chosen K
    best_k = suggestion.get('consensus_k') or suggestion.get('silhouette_best') or suggestion.get('elbow_best')
    if best_k is not None and best_k in k_vals:
        for axi in axes.flat:
            axi.axvline(int(best_k), color='red', linestyle='--', alpha=0.6, label=f'Chosen K={best_k}')
            axi.legend(loc='best', fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(results_dir, 'clustering', 'plots', 'k_selection_metrics.png')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    LOGGER.info("Saved K-selection chart -> %s", out_path)
    return out_path


def _pca_reduce_2d(X: np.memmap, sample_indices: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    from sklearn.decomposition import IncrementalPCA
    # IncrementalPCA for large arrays
    ipca = IncrementalPCA(n_components=2)
    # First pass: partial_fit
    for i in range(0, len(sample_indices), batch_size):
        idx = sample_indices[i : i + batch_size]
        ipca.partial_fit(np.asarray(X[idx]))
    # Second pass: transform
    Y = np.empty((len(sample_indices), 2), dtype=float)
    pos = 0
    for i in range(0, len(sample_indices), batch_size):
        idx = sample_indices[i : i + batch_size]
        Y[pos : pos + len(idx)] = ipca.transform(np.asarray(X[idx]))
        pos += len(idx)
    return Y


def plot_pca_scatter(results_dir: str, sample_size: int = 20000, random_state: int = 0) -> Optional[Tuple[str, str]]:
    plt = _safe_mpl_import()
    emb_path = os.path.join(results_dir, 'regex_embeddings.npy')
    labels_path = os.path.join(results_dir, 'clustering', 'labels.npy')
    if not os.path.exists(emb_path) or not os.path.exists(labels_path):
        LOGGER.warning("Missing embeddings or labels: %s | %s", emb_path, labels_path)
        return None

    X = np.load(emb_path, mmap_mode='r')  # type: ignore
    labels = np.load(labels_path)
    n = X.shape[0]
    if n != len(labels):
        LOGGER.warning("Embeddings and labels size mismatch: %d vs %d", n, len(labels))
        return None

    rng = np.random.RandomState(random_state)
    if sample_size <= 0 or sample_size >= n:
        sample_idx = np.arange(n)
    else:
        sample_idx = rng.choice(n, size=sample_size, replace=False)

    Y = _pca_reduce_2d(X, sample_idx)
    lbl = labels[sample_idx]

    # Scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=lbl, s=6, cmap='tab20', alpha=0.7)
    ax.set_title('PCA scatter of clustered embeddings')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    # Optional: colorbar for cluster ids (may be dense for big K)
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cluster ID')
    out_png = os.path.join(results_dir, 'clustering', 'plots', 'clusters_pca_scatter.png')
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    # Also save the 2D points for interactive tools if needed
    out_csv = os.path.join(results_dir, 'clustering', 'plots', 'clusters_pca_scatter.csv')
    np.savetxt(out_csv, np.column_stack([Y, lbl]), delimiter=',', header='pc1,pc2,cluster_id', comments='')

    LOGGER.info("Saved PCA scatter -> %s and coordinates -> %s", out_png, out_csv)
    return out_png, out_csv


def main_visualize(config_path: str, sample_size: int = 20000) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    cfg = _load_project_config(config_path)
    results_dir = cfg['paths']['output_dir']
    plots_dir = _ensure_plots_dir(results_dir)
    LOGGER.info("Saving plots under %s", plots_dir)

    plot_k_evaluations(results_dir)
    plot_pca_scatter(results_dir, sample_size=sample_size)


