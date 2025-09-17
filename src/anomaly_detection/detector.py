# src/anomaly_detection/detector.py
"""
Handles the anomaly detection (inference) workflow. Loads a pre-trained
model and uses it to find anomalies in new log data.
"""

import json
import logging
import os
from typing import Any, Dict

import numpy as np
from scipy.spatial.distance import cdist

# These imports assume your project structure is correct
from src.log_parser.parser import RawLogParser
from src.sentence_embedding.text_transformers import SentenceTransformerPipeline
from src.template_miner.log_templating import Drain3Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


class LogAnomalyDetector:
    """
    Detects anomalies in new log data using a pre-trained clustering model.
    """

    def __init__(self, model_summary_path: str):
        """
        Initializes the detector by loading trained model artifacts.

        Args:
            model_summary_path: Path to the model_summary.json file.
        """
        if not os.path.exists(model_summary_path):
            raise FileNotFoundError(f"Model summary file not found: {model_summary_path}")

        # Fix 1 : Common distance threshold config
        """
        logging.info(f"Initializing Log Anomaly Detector from {model_summary_path}...")
        with open(model_summary_path, "r") as f:
            summary = json.load(f)
            # --- FIX: Read the correct and newly available keys ---
            self.distance_threshold = summary["distance_threshold"]

            self.k = summary["k"]
            self.cluster_centers = np.load(summary["cluster_centers_file"])

        logging.info(
            f"Loaded model with k={self.k} and threshold={self.distance_threshold:.4f}"
        )
        """

        # Fix 2 : Per cluster distance threshold config
        with open(model_summary_path, "r") as f:
            summary = json.load(f)
            # --- FIX: Load the dictionary of thresholds ---
            self.per_cluster_thresholds = {
                int(k): v for k, v in summary["per_cluster_thresholds"].items()
            }
            self.k = summary["k"]
            self.cluster_centers = np.load(summary["cluster_centers_file"])
        logging.info(f"Loaded model with k={self.k} and per-cluster thresholds.")

    def find_anomalies(
            self, embeddings: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Finds anomalies in a new set of log embeddings.

        Args:
            embeddings: The new log embeddings to analyze. Assumes they are pre-normalized.

        Returns:
            A tuple containing:
            - An array of indices for the anomalous logs.
            - An array of distances for all input logs.
        """

        # Common distance_threshold for identifying the outliers
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Note: Normalization is now handled by the embedding pipeline, so it's removed here.
        distances = cdist(embeddings, self.cluster_centers).min(axis=1)
        outlier_indices = np.where(distances > self.distance_threshold)[0]
        return outlier_indices, distances
        """

        # Per-Cluster basis distance threshold checking
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Step 1: Find the closest cluster for each point
        distance_matrix = cdist(embeddings, self.cluster_centers)
        closest_cluster_indices = np.argmin(distance_matrix, axis=1)

        # Step 2: Get the actual distance to that closest cluster
        distances = distance_matrix[np.arange(len(embeddings)), closest_cluster_indices]

        # Step 3: Get the specific threshold for each point's assigned cluster
        thresholds_for_points = np.array(
            [self.per_cluster_thresholds.get(idx, float('inf')) for idx in closest_cluster_indices]
        )

        # Step 4: An anomaly is a point whose distance is greater than its own cluster's threshold
        outlier_indices = np.where(distances > thresholds_for_points)[0]

        return outlier_indices, distances


def detect_anomalies_in_file(config: Dict[str, Any]) -> None:
    """
    Full pipeline to take a new log file and detect anomalies in it.

    Args:
        config: The project configuration dictionary.
    """
    paths = config["paths"]
    log_file_path = paths["detection_log_file"]
    logging.info(f"\n--- Starting Anomaly Detection Pipeline for {log_file_path} ---")

    # 1. Parse the new log file
    parser = RawLogParser(config)
    log_df = parser.parse_new_log_file(log_file_path)
    if log_df.empty:
        logging.warning("No logs found in the file. Halting detection.")
        return

    # 2. Generate log templates using a pre-trained Drain model
    drain_pipeline = Drain3Pipeline(config)
    df_with_templates = drain_pipeline.predict(log_df, "message")

    # 3. Generate embeddings
    transformer_pipeline = SentenceTransformerPipeline(config)
    unique_templates = df_with_templates["template"].dropna().unique()
    template_embeddings = transformer_pipeline.embed_sentences(list(unique_templates))
    embedding_map = {
        template: emb for template, emb in zip(unique_templates, template_embeddings)
    }
    df_with_templates["embedding"] = df_with_templates["template"].map(embedding_map)
    df_to_analyze = df_with_templates.dropna(subset=["embedding"]).copy()

    # Save new logs df to file
    detection_logs_template_file = os.path.join(
        paths["output_dir"], paths["detection_logs_template_file"]
    )
    df_to_analyze.to_json(detection_logs_template_file, orient="records", lines=True)

    if df_to_analyze.empty:
        logging.warning("No valid embeddings could be generated for the logs.")
        return
    log_embeddings = np.vstack(df_to_analyze["embedding"].to_numpy())

    # 4. Detect anomalies
    output_summary_path = os.path.join(
        paths["output_dir"], paths["model_summary_file"]
    )
    detector = LogAnomalyDetector(output_summary_path)
    anomalous_indices, distances = detector.find_anomalies(log_embeddings)

    logging.info("\n--- Anomaly Detection Report ---")
    if len(anomalous_indices) > 0:
        logging.info(f"Found {len(anomalous_indices)} potential anomalies.")
        anomalous_logs = df_to_analyze.iloc[anomalous_indices].copy()
        anomalous_logs["distance"] = distances[anomalous_indices]

        logging.info("Anomalous Logs DataFrame columns: %s", list(anomalous_logs.columns))
        logging.info("Num Log: %d", len(anomalous_logs))

        # Save anomaly report to file
        anomaly_output_file = os.path.join(
            paths["output_dir"], paths["anomaly_output_file"]
        )
        anomalous_logs.to_csv(anomaly_output_file, index=False)
        logging.info("Saved anomaly output report to %s", anomaly_output_file)
    else:
        logging.info("No anomalies detected.")
