import logging
import os
import time
from contextlib import contextmanager
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

# Drain3 imports (install drain3)
try:
    from drain3 import TemplateMiner
    from drain3.file_persistence import FilePersistence
    from drain3.memory_buffer_persistence import MemoryBufferPersistence
    from drain3.template_miner_config import TemplateMinerConfig
except Exception as e:
    TemplateMiner = None  # type: ignore

from src.log_parser.parser import load_config, parse_logs_to_df
from src.template_miner.template_correction import correct_oracle_template

LOGGER = logging.getLogger(__name__)


class Drain3Pipeline:
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.paths = config['paths']
        self.drain_cfg = config.get('drain', {})
        self._tm = None

    def _make_template_miner(self):
        if TemplateMiner is None:
            raise RuntimeError("drain3 is not installed in this environment.")
        # Create/override config
        cfg = TemplateMinerConfig()

        # TODO: In PROD setup, we need to load drain3.ini and override values.
        sim = self.drain_cfg.get('similarity_threshold')
        if sim is not None:
            try:
                cfg.drain_sim_th = float(sim)
            except Exception:
                pass
        depth = self.drain_cfg.get('depth')
        if depth is not None:
            try:
                cfg.drain_depth = int(depth)
            except Exception:
                pass
        itv = self.drain_cfg.get('snapshot_interval_minutes')
        if itv is not None:
            try:
                cfg.snapshot_interval_minutes = int(itv)
            except Exception:
                pass
        comp = self.drain_cfg.get('compress_state')
        if comp is not None:
            cfg.compress_state = bool(comp)

        state_path = os.path.join(self.paths['output_dir'], self.paths['drain_state_file'])
        os.makedirs(os.path.dirname(state_path), exist_ok=True)

        tm = TemplateMiner(persistence_handler=FilePersistence(state_path), config=cfg)

        # Attempt to load previous state (safe on first run)
        try:
            tm.load_state()
        except Exception as e:
            LOGGER.info("No prior state to load: %s", e)
        return tm

    @property
    def tm(self):
        if self._tm is None:
            self._tm = self._make_template_miner()
        return self._tm

    def save_state(self, reason: str = "manual") -> None:
        try:
            self.tm.save_state(snapshot_reason=reason)
        except Exception as e:
            LOGGER.warning("Failed to save drain3 state: %s", e)

    def train_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Feed messages to Drain3 and return a DataFrame of cluster assignments."""

        results: List[Dict] = []
        with throttled_snapshots(self.tm, min_interval_sec=300):  # at most one save every 5 minutes
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Drain3 training"):
                msg = str(row['message']) if 'message' in row else str(row.get('raw', ''))
                if not msg:
                    continue
                try:
                    r = self.tm.add_log_message(msg)  # training: create/update clusters
                except Exception as e:
                    LOGGER.warning("Drain3 error on row %s: %s", idx, e)
                    continue
                # r is a dict including 'cluster_id', 'template_mined', 'template_id' (depends on version)
                # We normalize
                cluster_id = r.get('cluster_id') or r.get('cluster_id_matched') or r.get('cluster').cluster_id if r.get(
                    'cluster') else None
                size = getattr(r.get('cluster'), "size", None)
                template = r.get('template_mined') or (r.get('cluster').get_template() if r.get('cluster') else None)
                corrected_template = correct_oracle_template(template)
                was_corrected = template != corrected_template

                results.append({
                    'cluster_id': cluster_id,
                    'size': size,
                    'template': template,
                    'corrected_template': corrected_template,
                    'was_corrected': was_corrected,
                    "component": row.get('component', ''),
                    "log_level": row.get('level', '')
                })

            return pd.DataFrame(results)

    def export_current_mined_templates(self) -> pd.DataFrame:
        """
        Export current templates from Drain3 as a DataFrame with columns:
        ['cluster_id', 'template', 'size'].
        Compatible with different Drain3 layouts.
        """
        rows = []

        # Find the clusters mapping across common Drain3 layouts
        clusters_map = None

        # 1) Newer TemplateMiner may have .clusters
        clusters_map = getattr(self.tm, "clusters", None)

        # 2) Or it's hanging off .drain.clusters
        if not clusters_map:
            drain = getattr(self.tm, "drain", None)
            clusters_map = getattr(drain, "clusters", None) if drain is not None else None

        if not clusters_map:
            # Return an empty frame with the expected columns instead of a generic empty df
            return pd.DataFrame(columns=["cluster_id", "template", "size"])

        for cid, cluster in clusters_map.mapping.items():
            try:
                # Prefer the method; fall back to attribute if needed
                if hasattr(cluster, "get_template"):
                    template = cluster.get_template()
                else:
                    template = getattr(cluster, "template", None)

                size = getattr(cluster, "size", None)
                # Some Drain3 variants store the id on the cluster itself
                cluster_id = getattr(cluster, "cluster_id", cid)
                corrected_template = correct_oracle_template(template)
                was_corrected = template != corrected_template

                rows.append(
                    {"cluster_id": cluster_id, "template": template, "corrected_template": corrected_template,
                     "was_corrected": was_corrected, "size": size}
                )
            except Exception as e:
                # Be tolerant to odd cluster objects
                logging.warning("Skipping a cluster due to error: %s", e)

        # Build a DataFrame with stable, named columns
        return pd.DataFrame.from_records(rows, columns=["cluster_id", "template", "corrected_template", "was_corrected", "size"])


@contextmanager
def throttled_snapshots(tm: TemplateMiner, min_interval_sec: int = 300):
    """
    Temporarily replace tm.save_state with a throttled version.
    Ensures one final save on exit.
    """
    real_save = tm.save_state
    last = {"t": 0.0}

    def _throttled_save(*args, **kwargs):
        now = time.time()
        if now - last["t"] >= min_interval_sec:
            real_save(*args, **kwargs)
            last["t"] = now
        # else: skip snapshot

    tm.save_state = _throttled_save
    try:
        yield
    finally:
        tm.save_state = real_save
        # always persist once at the end
        tm.save_state(snapshot_reason="end_of_training")


def main(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
    config = load_config(config_path)

    # 1) Parse logs
    LOGGER.info("Parsing logs -> DataFrame...")
    log_df = parse_logs_to_df(config)

    logging.info("Log DataFrame columns: %s", list(log_df.columns))
    logging.info("Log head:\n%s", log_df.head(5).to_string())
    logging.info("Num Log: %d", len(log_df))

    out_dir = config['paths']['output_dir']
    os.makedirs(out_dir, exist_ok=True)

    # 2) Train Drain3
    pipeline = Drain3Pipeline(config)
    try:
        trained_logs_template_df = pipeline.train_from_df(log_df)

        # 3) Export templates
        tmpl_df = pipeline.export_current_mined_templates()

        logging.info("Templates DataFrame columns: %s", list(tmpl_df.columns))
        logging.info("Templates head:\n%s", tmpl_df.head(5).to_string())
        logging.info("Num templates: %d", len(tmpl_df))

        # 4) Persist
        corrected_jsonl = os.path.join(out_dir, config['paths']['corrected_templates_jsonl'])
        tmpl_df.to_json(corrected_jsonl, orient='records', lines=True)

        LOGGER.info("Exported %d templates (corrected) to %s", len(tmpl_df), corrected_jsonl)
    finally:
        pipeline.save_state(reason="end_of_run")


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='../configs/project.yaml')
    args = ap.parse_args()
    main(args.config)
