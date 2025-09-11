# Log Templating and Anamoly Detection Pipeline (Drain3)

A robust, production‑oriented skeleton for parsing logs, mining templates with Drain3,
post‑processing templates, and generating embeddings for search/clustering.

## Layout
- `src/configs/project.yaml` — runtime configuration
- `src/log_parser/parser.py` — file discovery and log parsing
- `src/template_miner/log_templating.py` — Drain3 training/inference pipeline
- `src/template_miner/template_correction.py` — post‑correction of oracle templates
- `src/sentence_embedding/text_transformers.py` — embedding generation

## Quick start
```bash
pip install drain3 sentence-transformers scikit-learn pandas pyyaml python-dateutil tqdm
python -m src.template_miner.log_templating --config src/configs/project.yaml
```

To build embeddings after mining:
```bash
python -m src.sentence_embedding.text_transformers --config src/configs/project.yaml
```
