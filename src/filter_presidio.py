"""
Anonymizador masivo con Presidio + HuggingFace Datasets
------------------------------------------------------------------
• sin copias intermedias     • paralelo con num_proc
• conserva orden y longitud  • inicialización perezosa de Presidio
"""

import argparse
import os
import time
from typing import Any, Dict, List

from datasets import Dataset, load_from_disk
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine


# -------------------------------------------------------------------
# Argumentos de línea de comandos
# -------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        required=False,
        help="Ruta del dataset (load_from_disk). Con --demo se ignora.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Ruta destino (save_to_disk)",
    )
    parser.add_argument(
        "--text_column",
        default="texto",
        help="Nombre de la columna de texto a anonimizar",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_proc", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--max_text_len", type=int, default=100_000)
    parser.add_argument(
        "--entities",
        default="EMAIL_ADDRESS,IP_ADDRESS,PHONE_NUMBER,CREDIT_CARD",
    )
    parser.add_argument("--language", default="en")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Crea un dataset de juguete y omite --input_path",
    )
    return parser.parse_args()


# -------------------------------------------------------------------
# Motores de análisis y anonimización (lazy init por proceso)
# -------------------------------------------------------------------
_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None


def _ensure_engines():
    """Inicializa motores en cada proceso (solo una vez por worker)."""
    global _analyzer, _anonymizer
    if _analyzer is None:
        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()


# -------------------------------------------------------------------
# Función que anonimiza un batch de textos
# -------------------------------------------------------------------
def _anonymize_batch(
    examples: Dict[str, Any],
    *,
    col: str,
    entities: List[str],
    language: str,
    max_len: int,
):
    """Aplica anonimización en lote, preservando orden original."""
    _ensure_engines()

    texts: List[str] = examples[col]
    valid_mask = [isinstance(t, str) and len(t) <= max_len for t in texts]
    valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]

    # Detectar entidades
    if valid_texts:
        if hasattr(_analyzer, "analyze_batch"):  # versiones modernas
            results = _analyzer.analyze_batch(
                text=valid_texts,
                language=language,
                entities=entities,
            )
        else:  # fallback a versiones antiguas
            results = [
                _analyzer.analyze(text=t, language=language, entities=entities)
                for t in valid_texts
            ]
    else:
        results = []

    # Aplicar anonimización y reconstruir orden original
    new_texts, idx = [], 0
    for t, ok in zip(texts, valid_mask):
        if ok:
            res = results[idx]
            idx += 1
            new_texts.append(
                _anonymizer.anonymize(text=t, analyzer_results=res).text if res else t
            )
        else:
            new_texts.append(t)

    examples[col] = new_texts
    return examples


# -------------------------------------------------------------------
# Dataset de demostración si se activa --demo
# -------------------------------------------------------------------
def build_demo_ds(col="texto") -> Dataset:
    demo = {
        col: [
            "Mi correo es juan.perez@gmail.com y mi IP es 192.168.1.1.",
            "Llama al +56 9 8765 4321",
            "Sin datos personales aquí.",
            "Tarjeta: 1234-5678-9876-5432; Email: user@dominio.com",
            "X" * 150_000,  # > max_len ⇒ queda sin tocar
        ]
    }
    return Dataset.from_dict(demo)


# -------------------------------------------------------------------
# Función principal
# -------------------------------------------------------------------
def main():
    args = parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

    entities = [e.strip() for e in args.entities.split(",") if e.strip()]
    t0 = time.perf_counter()

    # 1. Cargar dataset
    ds = build_demo_ds(args.text_column) if args.demo else load_from_disk(
        args.input_path
    )
    print(f"➡︎ Dataset original: {ds.num_rows:,} filas")

    # 2. Anonimizar en paralelo
    ds = ds.map(
        _anonymize_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc="🔐 Anonimizando",
        fn_kwargs=dict(
            col=args.text_column,
            entities=entities,
            language=args.language,
            max_len=args.max_text_len,
        ),
        load_from_cache_file=False,
        writer_batch_size=args.batch_size,
        keep_in_memory=False,
    )

    # 3. Guardar resultado
    print("💾 Guardando…")
    ds.save_to_disk(args.output_path)
    print(f"✅ Listo → {args.output_path}  (t={time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()
