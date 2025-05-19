# Latamgpt – Anonymization Filter

Utilities for **batch anonymization** of large-scale text datasets (Spanish, Portuguese, English).  
Combines [Microsoft Presidio] PII detection with custom regular-expressions for Latin-American IDs.

---

## 📁 Folder layout

```
Latamgpt/
├─ anonymization-filter/
│  ├─ requirements.txt
│  ├─ README.md             ← this file
│  ├─ .flake8  ·  .gitignore  ·  .pre-commit-config.yaml
│  └─ src/
│     ├─ filter_ID.py           # Only IDs  → <ID>
│     ├─ filter_presidio.py     # Only PII (e-mail, IP …)
│     └─ full_anon.py           # Full pipeline (PII + IDs)
└─ …
```

---

## 1 · Requirements

| Package                    | Min version | Notes                             |
| -------------------------- | ----------- | --------------------------------- |
| Python                     | 3.9         | tested 3.9 – 3.11                 |
| `datasets`                 | 2.19        | `load_from_disk` / `save_to_disk` |
| `presidio-analyzer`        | 2.2         | PII detection                     |
| `presidio-anonymizer`      | 2.2         | PII masking                       |
| `spacy` + `en_core_web_lg` | 3.x         | language model for Presidio       |
| `tqdm`                     | —           | progress bars                     |

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

---

## 2 · Scripts

### 2.1 `filter_ID.py`

_Replaces Latin-American identifiers (RUT, CURP, CPF, CUIT, …) with `<ID>`._

```bash
python src/filter_ID.py   --input_path  /data/ds_orig   --output_path /data/ds_ids   --column      texto
```

Use `--demo` to run a built-in test set.

---

### 2.2 `filter_presidio.py`

_PII anonymization only (e-mail, IP, phone, credit-card, …)._

```bash
python src/filter_presidio.py   --input_path  /data/ds_orig   --output_path /data/ds_pii   --text_column mensaje   --entities    EMAIL_ADDRESS,IP_ADDRESS,PHONE_NUMBER,CREDIT_CARD   --batch_size  32   --num_proc    8
```

`--text_column` accepts any single column name.

---

### 2.3 `full_anon.py`

**One-shot pipeline**:

1. Presidio → `<EMAIL_ADDRESS>`, `<PHONE_NUMBER>`, …
2. Regex IDs → `<ID>`

```bash
python src/full_anon.py   --input_path  /data/ds_orig   --output_path /data/ds_anon   --column      texto   --batch_size  64   --num_proc    8
```

#### Quick demo

```bash
python src/full_anon.py --demo --output_path ./demo_ds
```

Creates a 15-row sample containing RUT, CURP, e-mail, phone, IP and credit-card examples.

Load the result:

```python
from datasets import load_from_disk
ds = load_from_disk("./demo_ds")
print(ds[:]["texto"])
```

---

## 3 · Common flags

| Flag             | Description                        | Default  |
| ---------------- | ---------------------------------- | -------- |
| `--batch_size`   | Batch size for `datasets.map`      | 64       |
| `--num_proc`     | Parallel workers (multiprocessing) | CPU // 2 |
| `--max_text_len` | Skip texts longer than this length | 100 000  |
| `--language`     | Language code passed to Presidio   | `en`     |
