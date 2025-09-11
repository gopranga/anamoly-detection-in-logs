import os
import re
import glob
import logging
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
from pathlib import PurePosixPath

import pandas as pd
import yaml
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

_SANITIZE_PREFIX = re.compile(
    r'^(?:'
    r'\ufeff'  # BOM char U+FEFF
    r'|\ufffd'  # replacement char U+FFFD
    r'|\u00ef\u00bb\u00bf'  # mis-decoded BOM "ï»¿" (EF BB BF seen as Latin-1)
    r'|\u00ef\u00bf\u00bd'  # mis-decoded U+FFFD "ï¿½" (EF BF BD seen as Latin-1)
    r')+'
)

# Zero-width / bidi chars that can break matching
_ZW_BIDI = re.compile(r'[\u200b\u200c\u200d\u2060\u200e\u200f]')


@dataclass
class ParsedLog:
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
    def __init__(self, input_path: str, pattern: str = "**/*.log", exclude_globs: Optional[List[str]] = None) -> None:
        self.input_path = input_path
        self.pattern = pattern
        # Normalize exclude patterns to POSIX-style for robust matching
        self.exclude_patterns: List[str] = [p.replace('\\', '/') for p in (exclude_globs or [])]

    def _posix(self, path: str) -> str:
        return path.replace('\\', '/')

    def _should_exclude(self, path: str) -> bool:
        if not self.exclude_patterns:
            return False
        # Match against both absolute and input-root-relative POSIX paths
        abs_posix = self._posix(os.path.abspath(path))
        try:
            rel = os.path.relpath(path, self.input_path)
        except Exception:
            rel = path
        rel_posix = self._posix(rel)
        for pat in self.exclude_patterns:
            try:
                if PurePosixPath(abs_posix).match(pat) or PurePosixPath(rel_posix).match(pat):
                    return True
            except Exception:
                # Be tolerant to odd patterns
                continue
        return False

    def iter_files(self) -> Iterator[str]:
        if os.path.isfile(self.input_path):
            if not self._should_exclude(self.input_path):
                yield self.input_path
            return
        for path in glob.glob(os.path.join(self.input_path, self.pattern), recursive=True):
            if os.path.isfile(path):
                if not self._should_exclude(path):
                    yield path

    def detect_text_encoding(self, path: str, sample_size: int = 65536) -> str:
        """
        Detect the encoding of a text file.
        Priority:
          1) BOMs (UTF-8-SIG, UTF-16/32 LE/BE)
          2) UTF-16 heuristic when no BOM (NUL-byte pattern)
          3) charset-normalizer (if available)
          4) Fallback to 'utf-8'
        Returns a Python codec name usable by open(..., encoding=<name>).
        """
        with open(path, "rb") as fh:
            raw = fh.read(sample_size)

        # --- BOMs ---
        if raw.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if raw.startswith(b"\xff\xfe\x00\x00"):
            return "utf-32-le"
        if raw.startswith(b"\x00\x00\xfe\xff"):
            return "utf-32-be"
        if raw.startswith(b"\xff\xfe"):
            # BOM present -> let the codec handle endianness & strip BOM
            return "utf-16"
        if raw.startswith(b"\xfe\xff"):
            return "utf-16"

        # --- Heuristic UTF-16 without BOM (look for NUL patterns) ---
        # If many NUL bytes and they appear predominantly at even or odd positions,
        # assume UTF-16 LE or BE respectively.
        if raw:
            nul_total = raw.count(0)
            if nul_total / len(raw) > 0.2:  # plenty of NULs -> very likely UTF-16
                even_nuls = sum(1 for i in range(0, len(raw), 2) if raw[i] == 0)
                odd_nuls = nul_total - even_nuls
                if even_nuls > odd_nuls:
                    return "utf-16-be"
                else:
                    return "utf-16-le"

        # --- charset-normalizer (optional) ---
        try:
            from charset_normalizer import from_bytes  # type: ignore
            probe = from_bytes(raw).best()
            if probe and probe.encoding:
                # normalize names like 'UTF-8-SIG'
                enc = probe.encoding.lower()
                if enc.replace("_", "-") in ("utf-8-sig", "utf_8_sig"):
                    return "utf-8-sig"
                return probe.encoding
        except Exception:
            pass

        # Default
        return "utf-8"

    def _sanitize_line(self, s: str) -> str:
        if not s:
            return s
        # Remove any sequence of BOM/FFFD (and their Latin-1 mis-decoded forms) at the very start
        s = _SANITIZE_PREFIX.sub('', s)
        # Remove stray BOMs that might appear mid-line (rare but seen in concatenated logs)
        s = s.replace('\ufeff', '')
        # Normalize NBSP to space (your logs often have NBSPs around fields)
        s = s.replace('\u00A0', ' ')
        # Strip zero-width/bidi
        s = _ZW_BIDI.sub('', s)
        return s

    def iter_lines(self):
        for f in self.iter_files():
            enc = None
            try:
                enc = self.detect_text_encoding(f)
                # Open with detected encoding; for UTF‑8 with BOM/UTF‑16 with BOM,
                # the BOM is consumed by the codec automatically.
                with open(f, "r", encoding=enc, errors="replace", newline="") as fh:
                    for i, line in enumerate(fh, start=1):
                        # normalize newlines; keep '\n' only
                        if line.endswith("\r\n"):
                            line = line[:-2] + "\n"
                        line = line.rstrip("\n")
                        yield f, i, line.strip()
                continue
            except UnicodeError:
                # Try a small fallback ladder
                for fallback in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "latin-1"):
                    if fallback == enc:
                        continue
                    try:
                        with open(f, "r", encoding=fallback, errors="replace", newline="") as fh:
                            for i, line in enumerate(fh, start=1):
                                if line.endswith("\r\n"):
                                    line = line[:-2] + "\n"
                                line = line.rstrip("\n")
                                yield f, i, self._sanitize_line(line.strip())
                        break
                    except UnicodeError:
                        continue
            except OSError as e:
                # File access error; skip but log if you want
                # LOGGER.warning("Failed to read %s: %s", f, e)
                continue


class RawLogParser:
    def __init__(self, cfg: Dict) -> None:
        self.cfg = cfg or {}
        self._compile_patterns()
        self._compile_multiline()

    def _compile_patterns(self) -> None:
        pats = self.cfg.get('timestamp_level_patterns', [])
        # DOTALL so (?P<msg>.*) captures across newlines for grouped blocks
        self._compiled: List[re.Pattern] = [re.compile(p, re.DOTALL) for p in pats]
        lvl_map = self.cfg.get('level_map', {}) or {}
        self.level_map = {k.lower(): v for k, v in lvl_map.items()}

    def _compile_multiline(self) -> None:
        ml = self.cfg.get('multiline', {}) or {}
        self.ml_enabled: bool = bool(ml.get('enabled', True))
        start_regexes = ml.get('start_regexes') or [
            r'^\[\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?\]',
            r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}',
        ]
        self._start_patterns: List[re.Pattern] = [re.compile(rx) for rx in start_regexes]
        self.join_with: str = ml.get('join_with', ' ')
        self.max_lines: int = int(ml.get('max_lines', 2000))

    # --- Multiline grouping state machine ---
    def is_start_of_entry(self, line: str) -> bool:
        if not self.ml_enabled:
            return True  # each line is an own entry
        for pat in self._start_patterns:
            if pat.search(line):
                return True
        return False

    def iter_entries(self, lines: Iterable[Tuple[str, int, str]]) -> Iterator[Tuple[str, int, str]]:
        """
        Group raw lines into message blocks. Yields (file, start_line_no, block_text).
        A block starts when is_start_of_entry(line) is True.
        """
        buf: List[str] = []
        f0: Optional[str] = None
        start_ln: Optional[int] = None

        def flush():
            nonlocal buf, f0, start_ln
            if buf:
                block = self.join_with.join(buf)
                yield f0, start_ln, block
                buf = []
                f0 = None
                start_ln = None

        last_file: Optional[str] = None

        for f, ln, line in lines:
            # If file changed, flush previous block
            if last_file is not None and f != last_file:
                # flush block from previous file
                yield from flush()
            last_file = f

            if self.is_start_of_entry(line):
                # new block boundary
                if buf:
                    # flush previous block
                    yield from flush()
                buf = [line]
                f0 = f
                start_ln = ln
            else:
                # continuation
                if not buf:
                    # orphan continuation; treat as a block on its own
                    buf = [line]
                    f0 = f
                    start_ln = ln
                else:
                    buf.append(line)

            # safety: avoid unbounded growth
            if self.max_lines and buf and len(buf) >= self.max_lines:
                yield from flush()

        # flush any tail
        yield from flush()

    # --- Parsing a single (possibly multiline) block ---
    def parse_block(self, file: str, start_line_no: int, block: str) -> ParsedLog:
        # Try known patterns
        for pat in self._compiled:
            m = pat.match(block)
            if m:
                gd = m.groupdict()

                timestamp = m.group('timestamp').strip() if 'timestamp' in gd else None
                level = m.group('level').upper().strip() if 'level' in gd else None
                component = m.group('component').strip() if 'component' in gd else None
                pid = int(m.group('pid')) if 'pid' in gd else -1
                tid = int(m.group('tid')) if 'tid' in gd else -1
                file_guid_prefix = m.group('file_guid_prefix').strip() if 'file_guid_prefix' in gd else None
                line_no = int(m.group('line_no')) if 'line_no' in gd else -1
                message = m.group('message').strip() if 'message' in gd else block

                return ParsedLog(file, block, 'parsed', timestamp, self._normalize_level(level),
                                 component, pid, tid, file_guid_prefix, line_no, message)

        # Fallback: treat as raw block; optional fuzzy timestamp if you want
        return ParsedLog(file, block, 'raw', None, None, None,
                         -1, -1, None, -1, block)

    def _normalize_level(self, lvl: Optional[str]) -> Optional[str]:
        if not lvl:
            return None
        # Strip e.g. "[info   ]" forms that might slip through
        cleaned = str(lvl).strip().strip('[]').strip()
        mapped = self.level_map.get(cleaned.lower())
        return mapped or cleaned.upper()


def load_config(path: str) -> Dict:
    with open(path, 'r', encoding='utf-8') as fh:
        return yaml.safe_load(fh)


def parse_logs_to_df(config: Dict) -> pd.DataFrame:
    paths = config['paths']
    loader = LogDataLoader(
        paths['input_path'],
        paths.get('log_file_pattern', '**/*.log'),
        exclude_globs=paths.get('exclude_file_list') or []
    )
    parser = RawLogParser(config.get('parser', {}))

    rows: List[ParsedLog] = []
    for f, start_ln, block in tqdm(parser.iter_entries(loader.iter_lines()), desc="Grouping & parsing logs"):
        rec = parser.parse_block(f, start_ln, block)
        rows.append(rec)

    df = pd.DataFrame([asdict(r) for r in rows])
    return df


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
    try:
        raise Exception
        # log_df.to_parquet(os.path.join(out_dir, 'parsed_logs.parquet'), index=False)
        # LOGGER.info("Saved parsed logs to %s", parsed_path)
    except Exception as e:
        LOGGER.error("Exception in parquet file creation. Error - " + str(e))

        csv_path = os.path.join(out_dir, 'parsed_logs.csv')
        log_df.to_csv(csv_path, index=False)
        LOGGER.info("Saved parsed logs to %s", csv_path)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='../configs/project.yaml')
    args = ap.parse_args()
    main(args.config)
