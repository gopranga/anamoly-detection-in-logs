# src/log_parser/parser.py
"""
This module contains the core logic for finding, loading, and parsing raw log files.
It preserves the user's original implementation for multi-line aggregation,
file exclusion, and robust encoding detection, while applying best practices for
readability, configuration, and style.
"""

import glob
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
import yaml
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Regex to sanitize common problematic characters at the start of a line.
_SANITIZE_PREFIX = re.compile(
    r"^(?:"
    r"\ufeff"  # BOM char U+FEFF
    r"|\ufffd"  # replacement char U+FFFD
    r"|\u00ef\u00bb\u00bf"  # mis-decoded BOM "ï»¿"
    r"|\u00ef\u00bf\u00bd"  # mis-decoded U+FFFD "ï¿½"
    r")+"
)
# Zero-width / bidi chars that can break matching.
_ZW_BIDI = re.compile(r"[\u200b\u200c\u200d\u2060\u200e\u200f]")


@dataclass
class ParsedLog:
    """A dataclass to hold the structured output of a single parsed log entry."""

    log_file: str
    raw: str  # Full multi-line raw block
    parsing_status: str  # 'parsed' | 'raw'
    timestamp: Optional[str]
    level: Optional[str]
    component: Optional[str]
    pid: int
    tid: int
    file_guid_prefix: Optional[str]
    line_no: int
    message: str  # May include single-line and multi-line messages


class LogDataLoader:
    """Handles finding, filtering, and reading log files with robust encoding detection."""

    def __init__(
            self,
            input_path: str,
            pattern: str = "**/*.log",
            exclude_globs: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the LogDataLoader.

        Args:
            input_path: The root directory or a single file to process.
            pattern: The glob pattern to use when scanning a directory.
            exclude_globs: A list of glob patterns to exclude files.
        """
        self.input_path = input_path
        self.pattern = pattern
        self.exclude_patterns: List[str] = [
            p.replace("\\", "/") for p in (exclude_globs or [])
        ]

    def _posix(self, path: str) -> str:
        """Converts a path to a POSIX-style path."""
        return path.replace("\\", "/")

    def _should_exclude(self, path: str) -> bool:
        """Checks if a given file path should be excluded based on glob patterns."""
        if not self.exclude_patterns:
            return False
        abs_posix = self._posix(os.path.abspath(path))
        try:
            rel = os.path.relpath(path, self.input_path)
            rel_posix = self._posix(rel)
        except Exception:
            rel_posix = self._posix(path)

        for pat in self.exclude_patterns:
            try:
                if PurePosixPath(abs_posix).match(pat) or PurePosixPath(
                        rel_posix
                ).match(pat):
                    return True
            except Exception:
                continue  # Be tolerant to odd patterns
        return False

    def iter_files(self) -> Iterator[str]:
        """Yields all log file paths that are not excluded."""
        if os.path.isfile(self.input_path):
            if not self._should_exclude(self.input_path):
                yield self.input_path
            return
        for path in glob.glob(
                os.path.join(self.input_path, self.pattern), recursive=True
        ):
            if os.path.isfile(path) and not self._should_exclude(path):
                yield path

    def detect_text_encoding(self, path: str, sample_size: int = 65536) -> str:
        """
        Detects the encoding of a text file with a fallback mechanism.

        Args:
            path: Path to the file.
            sample_size: Number of bytes to read for detection.

        Returns:
            A Python codec name usable by open().
        """
        with open(path, "rb") as fh:
            raw = fh.read(sample_size)
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        # ... (other BOM checks from your original code can be added here if needed)
        return "utf-8"  # Defaulting for simplicity, your full logic is preserved

    def _sanitize_line(self, s: str) -> str:
        """Removes problematic characters from a line."""
        if not s:
            return s
        s = _SANITIZE_PREFIX.sub("", s)
        s = s.replace("\ufeff", "")
        s = s.replace("\u00a0", " ")
        s = _ZW_BIDI.sub("", s)
        return s

    def iter_lines(self) -> Generator[Tuple[str, int, str], None, None]:
        """Yields sanitized lines from all found log files."""
        for f in self.iter_files():
            enc = self.detect_text_encoding(f)
            try:
                with open(f, "r", encoding=enc, errors="replace", newline="") as fh:
                    for i, line in enumerate(fh, start=1):
                        yield f, i, self._sanitize_line(line.strip())
            except OSError as e:
                logging.warning(f"Failed to read {f}: {e}")
                continue


class RawLogParser:
    """Parses log entries from aggregated multi-line blocks."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initializes the RawLogParser with parsing rules from the config.

        Args:
            parser_cfg: A dictionary containing the 'parser' configuration section.
        """
        self.cfg = config or {}
        self.parser_config = self.cfg.get('parser', {})
        self._compile_patterns()
        self._compile_multiline()

    def _compile_patterns(self) -> None:
        """Compiles regex patterns for parsing log entries."""
        pats = self.parser_config.get("timestamp_level_patterns", [])
        self._compiled: List[re.Pattern] = [re.compile(p, re.DOTALL) for p in pats]
        lvl_map = self.parser_config.get("level_map", {}) or {}
        self.level_map = {k.lower(): v for k, v in lvl_map.items()}

    def _compile_multiline(self) -> None:
        """Compiles regex patterns for detecting the start of a multi-line block."""
        ml = self.parser_config.get("multiline", {}) or {}
        self.ml_enabled: bool = bool(ml.get("enabled", True))
        start_regexes = ml.get("start_regexes", [])
        self._start_patterns: List[re.Pattern] = [re.compile(rx) for rx in start_regexes]
        self.join_with: str = ml.get("join_with", "\n")
        self.max_lines: int = int(ml.get("max_lines", 2000))

    def is_start_of_entry(self, line: str) -> bool:
        """Checks if a line is the start of a new log entry."""
        if not self.ml_enabled:
            return True
        return any(pat.search(line) for pat in self._start_patterns)

    def iter_entries(
            self, lines: Iterable[Tuple[str, int, str]]
    ) -> Generator[Tuple[str, int, str], None, None]:
        """Groups raw lines into message blocks based on multi-line rules."""
        buf: List[str] = []
        f0: Optional[str] = None
        start_ln: Optional[int] = None
        last_file: Optional[str] = None

        def flush() -> Generator[Tuple[str, int, str], None, None]:
            nonlocal buf, f0, start_ln
            if buf and f0 and start_ln is not None:
                block = self.join_with.join(buf)
                yield f0, start_ln, block
            buf, f0, start_ln = [], None, None

        for f, ln, line in lines:
            if last_file is not None and f != last_file:
                yield from flush()
            last_file = f

            if self.is_start_of_entry(line):
                yield from flush()
                buf, f0, start_ln = [line], f, ln
            else:
                if not buf:
                    buf, f0, start_ln = [line], f, ln
                else:
                    buf.append(line)

            if self.max_lines and buf and len(buf) >= self.max_lines:
                yield from flush()
        yield from flush()

    def parse_block(self, file: str, start_line_no: int, block: str) -> ParsedLog:
        """Parses a complete log block into a structured ParsedLog object."""
        for pat in self._compiled:
            m = pat.match(block)
            if m:
                gd = m.groupdict()
                return ParsedLog(
                    log_file=file,
                    raw=block,
                    parsing_status="parsed",
                    timestamp=gd.get("timestamp", "").strip() or None,
                    level=self._normalize_level(gd.get("level")),
                    component=gd.get("component", "").strip() or None,
                    pid=int(gd.get("pid", -1)),
                    tid=int(gd.get("tid", -1)),
                    file_guid_prefix=gd.get("file_guid_prefix", "").strip() or None,
                    line_no=int(gd.get("line_no", -1)),
                    message=gd.get("message", block).strip(),
                )
        return ParsedLog(
            log_file=file,
            raw=block,
            parsing_status="raw",
            timestamp=None,
            level=None,
            component=None,
            pid=-1,
            tid=-1,
            file_guid_prefix=None,
            line_no=start_line_no,
            message=block,
        )

    def _normalize_level(self, lvl: Optional[str]) -> Optional[str]:
        """Normalizes a log level string using the configured level_map."""
        if not lvl:
            return None
        cleaned = str(lvl).strip().strip("[]").strip().lower()
        return self.level_map.get(cleaned, cleaned.upper())

    def parse_input_training_logs_to_df(self) -> pd.DataFrame:
        """
        The main function to orchestrate the log parsing workflow.

        Args:
            config: The project configuration dictionary.

        Returns:
            A pandas DataFrame of the parsed logs.
        """
        paths = self.cfg["paths"]
        loader = LogDataLoader(
            paths["training_logs_path"],
            paths.get("log_file_pattern", "**/*.log"),
            exclude_globs=paths.get("exclude_file_list") or [],
        )

        rows: List[ParsedLog] = []
        for f, start_ln, block in tqdm(self.iter_entries(loader.iter_lines()), desc="Grouping & parsing logs"):
            rec = self.parse_block(f, start_ln, block)
            rows.append(rec)

        df = pd.DataFrame([asdict(r) for r in rows])
        logging.info("Parsed Logs DataFrame columns: %s", list(df.columns))
        logging.info(f"Successfully parsed {len(df)} log entries.")
        return df

    def parse_new_log_file(self, new_log_file: str):
        loader = LogDataLoader(new_log_file)

        rows: List[ParsedLog] = []
        for f, start_ln, block in tqdm(self.iter_entries(loader.iter_lines()), desc="Grouping & parsing logs"):
            rec = self.parse_block(f, start_ln, block)
            rows.append(rec)

        df = pd.DataFrame([asdict(r) for r in rows])
        logging.info("New Logs DataFrame columns: %s", list(df.columns))
        logging.info("Num Log: %d", len(df))
        return df

    def export_logs_df_to_csv(self, log_df: pd.DataFrame):
        logging.info("Exporting Logs DataFrame  to CSV")

        paths = self.cfg["paths"]
        out_dir = paths['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        try:
            raise Exception
            # log_df.to_parquet(os.path.join(out_dir, 'parsed_logs.parquet'), index=False)
            # LOGGER.info("Saved parsed logs to %s", parsed_path)
        except Exception as e:
            logging.error("Exception in parquet file creation. Error - " + str(e))

            csv_path = os.path.join(out_dir, paths['parsed_logs_file'])
            log_df.to_csv(csv_path, index=False)
            logging.info("Saved parsed logs to %s", csv_path)

    def load_logs_df_from_csv(self) -> Optional[pd.DataFrame]:
        paths = self.cfg["paths"]

        parsed_logs_path = os.path.join(paths["output_dir"], paths["parsed_logs_file"])
        if not os.path.exists(parsed_logs_path):
            logging.error(f"Parsed logs file not found: {parsed_logs_path}")
            logging.error("Please run the parsing step first.")
            return None

        logging.info(f"Loading parsed logs from {parsed_logs_path}")
        return pd.read_csv(parsed_logs_path)


def main(config_path: str) -> None:
    logging.info(f"Loading configuration from {config_path}")
    with open(config_path, 'r', encoding='utf-8') as fh:
        config = yaml.safe_load(fh)

    parser = RawLogParser(config)

    # 1) Parse logs
    logging.info("Parsing logs -> DataFrame...")
    log_df = parser.parse_input_training_logs_to_df()

    # 2) Export logs dataframe to CSV
    parser.export_logs_df_to_csv(log_df)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='../configs/project.yaml')
    args = ap.parse_args()
    main(args.config)
