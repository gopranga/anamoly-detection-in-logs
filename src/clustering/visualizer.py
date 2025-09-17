# src/clustering/visualizer.py
"""
Provides functionality to visualize high-dimensional log embedding clusters
by reducing them to 2D using PCA and t-SNE, and then generating scatter plots.
"""

import logging
import os
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def create_cluster_visualizations(config: Dict[str, Any]) -> None:
    """
    Loads embeddings and cluster labels, performs dimensionality reduction,
    and saves scatter plot visualizations of the clusters.

    Args:
        config: The project configuration dictionary.
    """
    paths = config["paths"]
    output_dir = paths["output_dir"]
    embeddings_path = os.path.join(output_dir, paths["output_embeddings_file"])
    labels_path = os.path.join(output_dir, paths["labels_file"])
    pca_plot_path = os.path.join(output_dir, paths["pca_plot_file"])
    tsne_plot_path = os.path.join(output_dir, paths["tsne_plot_file"])

    # --- 1. Load the required data ---
    logging.info("Loading embeddings and cluster labels...")
    if not os.path.exists(embeddings_path) or not os.path.exists(labels_path):
        logging.error(
            f"Embeddings file ({embeddings_path}) or labels file ({labels_path}) not found. "
            "Please run the 'train' command first."
        )
        return

    embeddings = np.load(embeddings_path)
    labels = np.load(labels_path)

    # --- 2. Perform Dimensionality Reduction ---
    logging.info("Performing PCA dimensionality reduction...")
    pca = PCA(n_components=2, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings)

    logging.info("Performing t-SNE dimensionality reduction...")
    # For large datasets, t-SNE can be slow. A sample might be used in production.
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter_without_progress=1000)
    embeddings_tsne = tsne.fit_transform(embeddings)

    # --- 3. Create a DataFrame for plotting ---
    plot_df = pd.DataFrame({
        "pca-one": embeddings_pca[:, 0],
        "pca-two": embeddings_pca[:, 1],
        "tsne-one": embeddings_tsne[:, 0],
        "tsne-two": embeddings_tsne[:, 1],
        "cluster": labels,
    })

    # --- 4. Generate and Save Plots ---
    logging.info("Generating and saving cluster plots...")

    # Create PCA Plot
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="pca-one",
        y="pca-two",
        hue="cluster",
        palette=sns.color_palette("hsv", n_colors=len(plot_df["cluster"].unique())),
        data=plot_df,
        legend="full",
        alpha=0.7,
    )
    plt.title("Log Embedding Clusters (PCA Projection)")
    plt.savefig(pca_plot_path)
    logging.info(f"PCA visualization saved to {pca_plot_path}")
    plt.close()

    # Create t-SNE Plot
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-one",
        y="tsne-two",
        hue="cluster",
        palette=sns.color_palette("hsv", n_colors=len(plot_df["cluster"].unique())),
        data=plot_df,
        legend="full",
        alpha=0.7,
    )
    plt.title("Log Embedding Clusters (t-SNE Projection)")
    plt.savefig(tsne_plot_path)
    logging.info(f"t-SNE visualization saved to {tsne_plot_path}")
    plt.close()

    logging.info("Visualization complete.")