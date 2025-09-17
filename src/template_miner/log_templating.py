# src/template_miner/log_templating.py
"""
Handles log template mining using the Drain3 algorithm. This module defines the
pipeline for training the template miner from parsed logs and exporting the
resulting templates. It preserves the original logic of using a file-based
persistence handler and a throttled snapshot mechanism.
"""

import logging
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Drain3 imports are handled with a try-except block for graceful failure.
try:
    from drain3 import TemplateMiner
    from drain3.file_persistence import FilePersistence
    from drain3.template_miner_config import TemplateMinerConfig
except ImportError:
    # Allows the module to be imported even if drain3 is not installed.
    TemplateMiner = None  # type: ignore

from src.template_miner.template_correction import correct_oracle_template


class Drain3Pipeline:
    """A pipeline for training a Drain3 model and exporting log templates."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the Drain3Pipeline.

        Args:
            config: The project configuration dictionary.
        """
        if TemplateMiner is None:
            raise RuntimeError("drain3 is not installed in this environment.")

        self.config = config
        self.paths = config["paths"]
        self.drain_cfg = config.get("drain_config", {})
        self.correction_cfg = config.get("template_correction", {})
        self._tm: TemplateMiner = self._make_template_miner()

    def _make_template_miner(self) -> TemplateMiner:
        """
        Creates and configures a TemplateMiner instance from the project config.

        Returns:
            A configured TemplateMiner instance.
        """
        cfg = TemplateMinerConfig()

        state_path = os.path.join(
            self.paths["output_dir"], self.paths["drain_state_file"]
        )
        os.makedirs(os.path.dirname(state_path), exist_ok=True)

        self. persistence_path = state_path

        persistence = FilePersistence(self. persistence_path)
        tm = TemplateMiner(persistence_handler=persistence, config=cfg)

        try:
            tm.load_state()
            logging.info(f"Loaded {len(tm.drain.clusters)} templates from prior state.")
        except Exception as e:
            logging.info(f"No prior Drain3 state to load: {e}")
        return tm

    def save_state(self, reason: str = "manual") -> None:
        """Saves the current state of the template miner."""
        try:
            self._tm.save_state(snapshot_reason=reason)
        except Exception as e:
            logging.warning(f"Failed to save drain3 state: {e}")

    def train_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Feeds messages to Drain3 and returns a DataFrame of training results.

        Args:
            df: The DataFrame containing parsed log data.

        Returns:
            A DataFrame with details for each processed log message, including
            the assigned cluster and corrected template.
        """
        results: List[Dict[str, Any]] = []
        user_strings = self.correction_cfg.get("user_strings")
        message_col = self.config.get("pipeline_options", {}).get("message_column", "message")

        interval = self.drain_cfg.get("throttled_snapshot_interval_sec", 300)
        with throttled_snapshots(self._tm, min_interval_sec=interval):
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Drain3 training"):
                msg = str(row.get(message_col) or row.get("raw", ""))
                if not msg:
                    continue
                try:
                    result = self._tm.add_log_message(msg)
                    cluster_id = result["cluster_id"]
                    template = result["template_mined"]
                    corrected_template = template # correct_oracle_template(template, user_strings)
                    results.append(
                        {
                            "cluster_id": cluster_id,
                            "template": template,
                            "corrected_template": corrected_template,
                            "was_corrected": template != corrected_template,
                            "component": row.get("component"),
                            "log_level": row.get("level"),
                        }
                    )
                except Exception as e:
                    logging.warning(f"Drain3 error on log '{msg[:100]}...': {e}")
                    continue
        return pd.DataFrame(results)

    def predict(self, df: pd.DataFrame, message_column: str = "message", log_file_column: str = "log_file") -> pd.DataFrame:
        """
        Matches log messages against a pre-trained Drain3 model to get templates.

        Args:
            df: The DataFrame containing new log messages.
            message_column: The name of the column with log messages.

        Returns:
            The input DataFrame with an added 'template' column.
        """
        if not os.path.exists(self.persistence_path):
            raise FileNotFoundError(
                f"Drain state file not found at {self.persistence_path}. "
                "Please train the model first."
            )

        logging.info(
            f"Loading drain state for prediction from {self.persistence_path}"
        )
        self._tm.load_state()

        templates: List[Optional[str]] = []
        df_new_log_messages = pd.DataFrame(columns=['log_file', 'new_log_message'])
        logging.info("Matching logs to existing templates...")

        delimiter = "___"
        index = 0
        for msg in tqdm((df[log_file_column].astype(str) + delimiter + df[message_column].astype(str)).tolist(), desc="Matching templates"):
            msg_parts = msg.split(delimiter, 2)
            log_file = msg_parts[0]
            raw_msg = msg_parts[1]

            # --- CRITICAL FIX STARTS HERE ---
            result = self._tm.match(raw_msg)
            if result:
                # A template was successfully matched
                templates.append(result.get_template())
            else:
                df_new_log_messages.loc[index, 'log_file'] = log_file
                df_new_log_messages.loc[index, 'new_log_message'] = raw_msg

                templates.append(raw_msg)
            # --- CRITICAL FIX ENDS HERE ---
            index += 1

        if not df_new_log_messages.empty:
            output_file_path = os.path.join(self.paths["output_dir"], self.paths["detection_new_log_messages_file"])
            df_new_log_messages.to_csv(output_file_path, index=False)
            logging.info(f"Saved new log message events in file={output_file_path}")
            logging.info(f"Num of new log message events = {len(df)}")

        df_with_templates = df.copy()
        df_with_templates["template"] = templates
        return df_with_templates

    def export_current_mined_templates(self) -> pd.DataFrame:
        """
        Exports current templates from Drain3 as a DataFrame.

        Returns:
            A DataFrame with columns: ['cluster_id', 'template', 'corrected_template',
            'was_corrected', 'size'].
        """
        clusters = self._tm.drain.clusters
        user_strings = self.correction_cfg.get("user_strings")
        rows = []
        for cluster in clusters:
            template = cluster.get_template()
            corrected_template = template # correct_oracle_template(template, user_strings)
            rows.append(
                {
                    "cluster_id": cluster.cluster_id,
                    "template": template,
                    "corrected_template": corrected_template,
                    "was_corrected": template != corrected_template,
                    "size": cluster.size,
                }
            )
        return pd.DataFrame(
            rows,
            columns=[
                "cluster_id",
                "template",
                "corrected_template",
                "was_corrected",
                "size",
            ],
        )


@contextmanager
def throttled_snapshots(
        tm: TemplateMiner, min_interval_sec: int = 300
) -> Generator[None, None, None]:
    """
    A context manager to temporarily replace tm.save_state with a throttled version,
    ensuring one final save on exit.

    Args:
        tm: The TemplateMiner instance.
        min_interval_sec: The minimum time in seconds between automatic snapshots.
    """
    real_save_state = tm.save_state
    last_snapshot_time = {"t": 0.0}

    def _throttled_save(*args: Any, **kwargs: Any) -> None:
        now = time.time()
        if now - last_snapshot_time["t"] >= min_interval_sec:
            real_save_state(*args, **kwargs)
            last_snapshot_time["t"] = now

    tm.save_state = _throttled_save
    try:
        yield
    finally:
        tm.save_state = real_save_state
        logging.info("Persisting final Drain3 state...")
        tm.save_state(snapshot_reason="end_of_training")


def main(config_path: str) -> None:
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. Load previously parsed logs
    # This step assumes a prior script (like parser.py) has run and produced this file.
    from src.log_parser.parser import RawLogParser

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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='../configs/project.yaml')
    args = ap.parse_args()
    main(args.config)
