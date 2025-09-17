# src/sentence_embedding/text_transformers.py
"""
Handles the conversion of text (log templates) into numerical vector embeddings.
This module can be run directly to test the embedding functionality, including
saving the generated embeddings to a file.
"""

import logging
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Configure logging for standalone execution and module usage.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)


class SentenceTransformerPipeline:
    """
    A pipeline for generating embeddings using a SentenceTransformer model.
    The model is loaded once upon initialization for efficiency.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the pipeline and loads the pre-trained model from config.

        Args:
            config: A dictionary containing the 'embedding_config'.
        """
        embedding_cfg = config.get("embedding_config", {})
        model_name = embedding_cfg.get("model_name", "all-MiniLM-L6-v2")

        logging.info(f"Loading SentenceTransformer model: {model_name}...")
        try:
            self.model = SentenceTransformer(model_name)
            logging.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logging.error(
                f"Failed to load SentenceTransformer model '{model_name}'. Error: {e}"
            )
            raise

    def embed_sentences(
            self, sentences: List[str], normalize_embeddings: bool = True
    ) -> np.ndarray:
        """
        Generates embeddings for a list of sentences.

        Args:
            sentences: A list of strings to be embedded.
            normalize_embeddings: If True, L2-normalizes the embeddings.

        Returns:
            A numpy array of the generated embeddings.
        """
        if not sentences:
            return np.array([])

        logging.info(f"Generating embeddings for {len(sentences)} sentences...")
        embeddings = self.model.encode(sentences, show_progress_bar=True)

        if normalize_embeddings:
            logging.info("Normalizing embeddings to unit length.")
            embeddings = normalize(embeddings, norm="l2", axis=1)

        return embeddings


def main(config_path: str) -> None:
    logging.info("--- Running Sentence Transformer Pipeline Test ---")

    # 1. Load configuration from the YAML file
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found at: {config_path}")
        return

    # 2. Initialize the pipeline
    try:
        pipeline = SentenceTransformerPipeline(config=config)
    except Exception as e:
        logging.error(
            "Failed to initialize pipeline. "
            "Ensure you have run 'pip install -r requirements.txt'. Error: %s",
            e,
        )
        return

    # 3. Log templates for embedding
    paths = config["paths"]

    corrected_templates_file_path = os.path.join(
        paths["output_dir"], paths["corrected_templates_file"]
    )
    df = pd.read_json(corrected_templates_file_path, orient='records', lines=True)
    sample_templates = df['corrected_template'].astype(str).tolist()
    logging.info("Embedding %d sample templates...", len(sample_templates))

    # 4. Generate embeddings
    embeddings = pipeline.embed_sentences(sample_templates)

    # 5. Save the embeddings to the output file specified in the config
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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='../configs/project.yaml')
    args = ap.parse_args()
    main(args.config)
