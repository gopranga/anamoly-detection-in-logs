# src/clustering/cli.py
"""
Command-Line Interface for the Log Anomaly Detection pipeline.
This script provides distinct commands to run each stage of the pipeline,
from parsing raw logs to detecting anomalies, all controlled by a central
YAML configuration file.
"""

import argparse
import logging
import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

from src.anomaly_detection.detector import detect_anomalies_in_file
from src.clustering.kmeans_runner import run_kmeans_pipeline
from src.clustering.visualizer import create_cluster_visualizations
from src.log_parser.parser import RawLogParser
from src.sentence_embedding.text_transformers import SentenceTransformerPipeline
from src.template_miner.log_templating import Drain3Pipeline

# Set up basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def handle_parse(config: Dict[str, Any]) -> None:
    """Orchestrates the log parsing workflow."""
    logging.info("--- Running Log Parsing Workflow ---")

    parser = RawLogParser(config)

    # 1) Parse logs
    logging.info("Parsing logs -> DataFrame...")
    log_df = parser.parse_input_training_logs_to_df()

    # 2) Export logs dataframe to CSV
    parser.export_logs_df_to_csv(log_df)


def handle_template(config: Dict[str, Any]) -> None:
    """Orchestrates the log templating workflow."""
    logging.info("--- Running Log Templating Workflow ---")

    # 1. Load previously parsed logs
    # This step assumes a prior script (like parser.py) has run and produced this file.
    parser = RawLogParser(config)
    log_df = parser.load_logs_df_from_csv()

    # 2. Train Drain3
    pipeline = Drain3Pipeline(config)
    try:
        pipeline.train_from_df(log_df)

        # 3. Export templates
        templates_df = pipeline.export_current_mined_templates()
        logging.info(f"Extracted {len(templates_df)} unique templates.")

        logging.info("Unique Drain3 mined templates columns: %s", list(templates_df.columns))
        logging.info("Num templates: %d", len(templates_df))

        # 4. Persist results
        paths = config["paths"]
        output_path = os.path.join(
            paths["output_dir"], paths["corrected_templates_file"]
        )
        templates_df.to_json(output_path, orient="records", lines=True)
        logging.info(f"Exported templates (with corrections) to {output_path}")

    finally:
        pipeline.save_state(reason="end_of_run")

    logging.info("Log templating complete. Drain3 state has been saved.")


def handle_embed(config: Dict[str, Any]) -> None:
    """Orchestrates the log embedding workflow."""
    logging.info("--- Running Log Embedding Workflow ---")

    # 1. Initialize the pipeline
    try:
        pipeline = SentenceTransformerPipeline(config=config)
    except Exception as e:
        logging.error(
            "Failed to initialize pipeline. "
            "Ensure you have run 'pip install -r requirements.txt'. Error: %s",
            e,
        )
        return

    # 2. Log templates for embedding
    paths = config["paths"]

    corrected_templates_file_path = os.path.join(
        paths["output_dir"], paths["corrected_templates_file"]
    )
    df = pd.read_json(corrected_templates_file_path, orient='records', lines=True)
    sample_templates = df['corrected_template'].astype(str).tolist()
    logging.info("Embedding %d sample templates...", len(sample_templates))

    # 3. Generate embeddings
    embeddings = pipeline.embed_sentences(sample_templates)

    # 4. Save the embeddings to the output file specified in the config
    if embeddings.size > 0:
        output_file = os.path.join(
            paths["output_dir"], paths["output_embeddings_file"]
        )

        if not output_file:
            logging.error("'output_embeddings_file' not defined in config paths.")
            return

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            np.save(output_file, embeddings)
            logging.info(f"Embeddings successfully saved to: {output_file}")
        except Exception as e:
            logging.error(f"Failed to save embeddings to {output_file}. Error: {e}")

        logging.info("\n--- Test Results ---")
        logging.info("Shape of embeddings matrix: %s", embeddings.shape)
        logging.info("First 5 dimensions of the first vector: %s", embeddings[0, :5])
    else:
        logging.warning("Failed to generate embeddings. Nothing to save.")


def main() -> None:
    """Parses command-line arguments and runs the appropriate workflow."""
    parser = argparse.ArgumentParser(description="Log Anomaly Detection CLI")
    parser.add_argument(
        "--config",
        default="../configs/project.yaml",
        help="Path to the project config file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Define a command for each stage of the pipeline
    subparsers.add_parser("parse", help="Parse raw log files into a structured format.")
    subparsers.add_parser("template", help="Mine log templates from parsed logs using Drain3.")
    subparsers.add_parser("embed", help="Generate numerical embeddings from log templates.")
    subparsers.add_parser("train", help="Train the anomaly detection model on baseline embeddings.")
    subparsers.add_parser("detect", help="Detect anomalies in a new log file using a trained model.")
    subparsers.add_parser("visualize", help="Generate 2D visualizations of the log clusters.")

    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {args.config}")
        return

    # Execute the chosen command
    if args.command == "parse":
        handle_parse(config)
    elif args.command == "template":
        handle_template(config)
    elif args.command == "embed":
        handle_embed(config)
    elif args.command == "train":
        logging.info("--- Running Full Training Workflow (K-Selection and Model Fit) ---")
        run_kmeans_pipeline(config)
    elif args.command == "detect":
        logging.info("--- Running Anomaly Detection Workflow ---")
        detect_anomalies_in_file(config)
    elif args.command == "visualize":
        logging.info("--- Running Cluster Visualization Workflow ---")
        create_cluster_visualizations(config)


if __name__ == "__main__":
    main()
