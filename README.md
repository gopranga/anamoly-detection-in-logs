# Log Templating and Anomaly Detection Pipeline (Drain3)

A robust, production‑oriented skeleton for parsing logs, mining templates using Drain3, post‑processing templates for clarity, embedding generation for clustering/search, and detecting anomalies in large-scale log data. It includes modular and extensible components for log analysis workflows.

## Features
- **Log Parsing:** Extracts structured data from raw logs, supporting multi-line aggregation and robust encoding detection.
- **Template Mining:** Mines log templates using Drain3 with persistence and incremental template correction.
- **Embedding Generation:** Converts templates into numerical vectors for downstream clustering tasks using pre-trained sentence transformers.
- **Clustering:** Supports evaluation of optimal cluster sizes (`k`) and training scalable clustering algorithms, including KMeans or MiniBatchKMeans.
- **Anomaly Detection:** Identifies anomalous logs using pre-trained clustering models, supporting both common and per-cluster distance thresholds.
- **Extensibility:** Config-driven architecture with YAML files for customization.

---

## Repository Layout

```
src/
├── anomaly_detection/    # Anomaly detection modules
│   └── detector.py       # Detects anomalies in log data using pre-trained clustering models
├── clustering/           # Clustering utilities and evaluation
│   ├── kmeans_runner.py  # KMeans training pipeline
│   ├── k_selection.py    # Optimal K evaluation for clustering
├── configs/              # YAML configuration files
│   └── project.yaml      # Central runtime configuration
├── log_parser/           # Log file parsing utilities
│   └── parser.py         # Multi-line log parsing with flexible regex patterns
├── sentence_embedding/   # Embedding generation for clustering and similarity search
│   └── text_transformers.py # Converts log templates into vector embeddings
├── template_miner/       # Log template mining and post-processing
│   ├── log_templating.py # Drain3 training and mining pipeline
│   └── template_correction.py # Corrects mined templates for consistency
```

---

## Quick Start

### 1. Install Dependencies
```bash
pip install drain3 sentence-transformers scikit-learn pandas pyyaml python-dateutil tqdm
```

### 2. Parse Logs into Structured Data
To parse raw logs into structured data, update your config in `src/configs/project.yaml`, then run:
```bash
python -m src.log_parser.parser --config src/configs/project.yaml
```

### 3. Train Log Template Miner (Drain3)
Mine templates from the parsed logs and persist the state:
```bash
python -m src.template_miner.log_templating --config src/configs/project.yaml
```

### 4. Generate Embeddings for Templates
Convert mined templates into numerical embeddings:
```bash
python -m src.sentence_embedding.text_transformers --config src/configs/project.yaml
```

### 5. Train KMeans Clustering Model
Evaluate optimal `k` and train the clustering algorithm:
```bash
python -m src.clustering.kmeans_runner --config src/configs/project.yaml
```

### 6. Detect Anomalies in New Log Data
To process new logs and detect anomalies:
```bash
python -m src.anomaly_detection.detector --config src/configs/project.yaml
```

---

## Modules Overview

### 1. Log Parsing (`parser.py`)
- **Purpose:** Extracts logs from files and parses them into structured data, supporting multi-line log entries.
- **Key Features:**
  - Multi-line log block aggregation using regex patterns.
  - Encoding detection and de-sanitization for robust processing.
  - Output: Parsed logs exported to CSV/Parquet for downstream processing.

### 2. Template Mining (`log_templating.py`)
- **Purpose:** Discovers and corrects templates from parsed log messages using the Drain3 algorithm.
- **Key Features:**
  - File-based persistence for checkpoints and state recovery.
  - Configurable snapshot throttling and incremental model updates.
  - Heuristic corrections applied to templates for better readability.

### 3. Template Correction (`template_correction.py`)
- **Purpose:** Heuristically normalizes mined log templates to a more human-readable format.
- **Key Features:**
  - Collapses redundant patterns (e.g., numbers, paths, and booleans) into placeholders.
  - Custom user-defined strings for normalization.

### 4. Embedding Generation (`text_transformers.py`)
- **Purpose:** Converts templates into vectorized embeddings using pre-trained models.
- **Key Features:**
  - Efficient embedding generation with optional L2 normalization.
  - Integrates with `sentence-transformers` library for flexibility.

### 5. KMeans Clustering (`kmeans_runner.py`)
- **Purpose:** Trains clustering models for grouping similar log patterns.
- **Key Features:**
  - Supports dynamic `k` evaluation and selection using multiple metrics (e.g., Silhouette, Elbow Method).
  - Memory-efficient streaming for large datasets.

### 6. Anomaly Detection (`detector.py`)
- **Purpose:** Identifies anomalies in newly observed logs using pre-trained clustering models.
- **Key Features:**
  - Common or per-cluster distance thresholds.
  - Full integration with parsing, mining, and embedding pipelines.

---

## Configuration

All runtime configurations are defined in `src/configs/project.yaml`. Key sections include:
- **Paths:** Input/output file paths for logs, embeddings, templates, etc.
- **Parsing Rules:** Regex patterns, level mappings, and multi-line log configurations.
- **Drain3 Configuration:** Snapshot intervals, persistence file paths, and training options.
- **Embedding Config:** Pre-trained model selection (e.g., `all-MiniLM-L6-v2`).
- **Clustering Options:** KMeans parameters, evaluation methods, and thresholds.
- **Anomaly Detection:** Output paths and model summaries for detection runs.

---

### Example Config (`project.yaml`)

```yaml
paths:
  training_logs_path: "data/raw_logs/"
  output_dir: "data/processed/"
  parsed_logs_file: "parsed_logs.csv"
  corrected_templates_file: "corrected_templates.json"
  output_embeddings_file: "embeddings.npy"
  model_summary_file: "model_summary.json"
  anomaly_output_file: "anomalies.csv"
parser:
  timestamp_level_patterns:
    - timestamp: "(?P<timestamp>\\d{4}-\\d{2}-\\d{2})"
  multiline:
    enabled: true
    start_regexes: ["^\\d{4}-\\d{2}-\\d{2}"]
drain_config:
  throttled_snapshot_interval_sec: 300
embedding_config:
  model_name: "all-MiniLM-L6-v2"
clustering:
  method: "kmeans"
  k_selection:
    start: 2
    stop: 50
    step: 2
    random_state: 42
anomaly_detection:
  distance_threshold: 95
```

---

## Future Enhancements
- Support for real-time log ingestion pipelines.
- Integration with advanced anomaly detection (e.g., Isolation Forest or Deep Learning).

---
