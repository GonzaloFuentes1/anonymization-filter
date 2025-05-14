# Anonymization Filter

This repository provides tools to detect and anonymize Personally Identifiable Information (PII) in text data using both regex-based methods and NVIDIA's NeMo-Curator library.

## Features

- Regex-based anonymization for Latin American identifiers (RUT, CUIT, CURP, etc.).
- Transformer-based anonymization using NeMo-Curator (CPU or CUDA).
- Parallel processing using Dask for large datasets.
- Benchmarking tools to compare performance across configurations.

## File Overview

```
anonymization-filter/
├── resultados_benchmark/       # Benchmark results and plots
├── .flake8                     # Linting rules
├── .gitignore
├── .pre-commit-config.yaml    # Pre-commit hooks
├── LICENSE
├── README.md                   # This file
├── requirements.txt           # Python dependencies
├── dataset_filter_ID.py       # Regex-based anonymization for datasets
├── dataset_filter_nemo.py     # NeMo-based anonymization for datasets
├── text_filter_ID.py          # Regex-based anonymization for text
├── text_filter_nemo.py        # NeMo-based anonymization for text
```

## Installation

```bash
pip install -r requirements.txt
```

To install `nemo-curator`, clone and install it manually (if not on PyPI):

```bash
pip install --extra-index-url https://pypi.nvidia.com nemo-curator[all]
```

## Usage

### Regex-Based Anonymization

Text file:

```bash
python text_filter_ID.py
```

Dataset (Hugging Face):

```bash
python dataset_filter_ID.py \
  --input_path /ruta/al/dataset/original \
  --output_path /ruta/al/dataset/anonimizado \
  --column text

```

### NeMo-Based Anonymization

Text file:

```bash
python text_filter_nemo.py
```

Dataset with full configuration:

```bash
python dataset_filter_nemo.py \
  --input_path input/path \
  --output_path output/path \
  --text_column texto \
  --chunk_size 10000 \
  --n_workers 32 \
  --threads_per_worker 1 \
  --device cpu \
  --supported_entities EMAIL_ADDRESS,PHONE_NUMBER,CREDIT_CARD,IP_ADDRESS
```

## Benchmarking

Run benchmarking suite:

```bash
python benchmark.py
```

This tests anonymization speed for different worker/thread combinations and chunk sizes. Plots and CSV summaries are stored in `resultados_benchmark/`.

## License

This project is licensed under the MIT License.
