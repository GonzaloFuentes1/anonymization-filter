"""
Anonimización completa (Presidio + IDs) para Hugging Face Datasets
-----------------------------------------------------------------
• Paso 1 – Presidio: EMAIL_ADDRESS, PHONE_NUMBER, … (configurable)
• Paso 2 – Regex: RUT, CURP, CPF, CUIT, etc.  →  <ID>
• Paralelizable con --num_proc, sin copias intermedias
"""

from __future__ import annotations

import argparse
import os
import re
import time
from typing import Any, Dict, List, Tuple

from datasets import Dataset, load_from_disk
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# --------------------------------------------------------------------------- #
# Patrones de identificadores (compilados una sola vez)                       #
# --------------------------------------------------------------------------- #
RAW_PATTERNS: Dict[str, str] = {
    "CI_NIC":  r"\b\d{3}[-\s]\d{6}[-\s]\d{4}[A-Z]?\b",
    "RUC_NIC": r"\b\d{3}[-\s]\d{6}[-\s]\d{4}[A-Z]?\b",
    "CPF_BRA": r"\b\d{3}[.\s-]\d{3}[.\s-]\d{3}[.\s-]\d{2}\b",
    "DPI_GTM": r"\b\d{4}\s\d{5}\s\d{4}\b",
    "CURP_MEX": r"\b[A-Z]{4}-?\d{6}-?[HM][A-Z]{5}[A-Z0-9]\b",
    "RFC_MEX": r"\b[A-ZÑ&]{3,4}-?\d{6}-?[A-Z0-9]{3}\b",
    "RIF_VEN": r"\b[JGVEP][- ]\d{8}[- ]\d\b",
    "CI_BOL": r"\b\d{6,8}[-\s][A-Z]{2}\b",
    "RUC_PRY": r"\b\d{6,8}[A-Z]?[-\s]\d\b",
    "CUIT_ARG": r"\b\d{2}[.\s-]\d{8}[.\s-]\d\b",
    "CI_URU": r"\b\d{1,2}[.\s-]\d{3}[.\s-]\d{3}[.\s-]\d\b",
    "RUT_CHI": r"\b\d{1,2}[.\s-]\d{3}[.\s-]\d{3}[.\s-]?[\dkK]\b",
    "CI_VEN": r"\b[VvEe][- ]\d{6,8}\b",
    "PAS_ARG": r"\bAA[-\s]\d{7}\b",
    "PAS_CHI": r"\b[Cc]-\d{8}\b",
    "PAS_MEX": r"\bG-\d{8}\b",
    "ID_HTI": r"\b\d{2}[-\s]\d{2}[-\s]\d{2}[-\s]\d{5}\b",
    "CI_CRI": r"\bCR[-\s]?\d[-\s]?\d{4}[-\s]?\d{4}\b",
    "CI_CUB": r"\bCUB[-\s]?\d{6}[-\s]?\d{5}\b",
    "NIT_BOL": r"\bBO[-\s]?\d{6,8}[-\s]?\d\b",
    "NIT_COL": r"\bCOL[-\s]?\d{8,10}-\d\b",
    "NIT_GTM": r"\bGT[-\s]?\d{6,8}[-\s]?\d\b",
    "NIT_SLV": r"\bSV[-\s]?\d{4}[-\s]?\d{6}[-\s]?\d{3}[-\s]?\d\b",
    "RNC_DOM": r"\bRD[-\s]?\d[-\s]?\d{2}[-\s]?\d{5}[-\s]?\d\b",
    "RTN_HND": r"\bHN[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{5}\b",
    "RUC_ECU": r"\bEC[-\s]?\d{10}[-\s]?\d{3}\b",
    "RUC_PAN": r"\bP[-\s]?\d{1,4}[-\s]?\d{1,4}[-\s]?\d{1,4}\b",
    "RUC_PER": r"\bPE[-\s]?(10|15|16|17|20)\d{8}\b",
}

PATTERNS: List[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE) for p in RAW_PATTERNS.values()
    ]


def replace_identifiers(text: str, label: str = "<ID>") -> str:
    """Reemplaza IDs sin permitir solapamientos (evita <ID<ID>)."""
    # 1. Recolectar todas las coincidencias (patrón, inicio, fin)
    spans: List[Tuple[int, int]] = []
    for pat in PATTERNS:
        spans.extend(m.span() for m in pat.finditer(text))

    if not spans:
        return text

    # 2. Ordenar: primero los más largos, luego por inicio
    spans.sort(key=lambda s: (-(s[1] - s[0]), s[0]))

    chars = list(text)
    occupied = [False] * len(chars)

    for start, end in spans:
        # ¿Algún carácter de este rango ya fue reemplazado?
        if any(occupied[start:end]):
            continue  # solapa → saltar

        repl = list(label.ljust(end - start))
        chars[start:end] = repl
        for i in range(start, end):
            occupied[i] = True

    return "".join(chars)


# --------------------------------------------------------------------------- #
# Presidio (lazy init por worker)                                             #
# --------------------------------------------------------------------------- #

_ANALYZER: AnalyzerEngine | None = None
_ANONYMIZER: AnonymizerEngine | None = None


def ensure_presidio() -> Tuple[AnalyzerEngine, AnonymizerEngine]:
    global _ANALYZER, _ANONYMIZER
    if _ANALYZER is None:
        _ANALYZER = AnalyzerEngine()
        _ANONYMIZER = AnonymizerEngine()
    return _ANALYZER, _ANONYMIZER


# --------------------------------------------------------------------------- #
# Dataset de demostración (la lista completa solicitada)                      #
# --------------------------------------------------------------------------- #
def build_demo_dataset(col: str = "texto") -> Dataset:
    """Dataset breve para pruebas rápidas (PII + IDs)."""
    data = {
        col: [
            # ── Identificadores latinoamericanos ───────────────────────────
            "Mi RUT es 12.345.678-9",                 # CL
            "CURP: GOML840512HDFRRN09",               # MX
            "RFC: GOM8405121A1",                      # MX
            "CI Bolivia: 12345678-LP",                # BO
            "CUIT: 20-12345678-1",                    # AR

            # ── PII estándar para Presidio ────────────────────────────────
            "Correo: juan.perez@example.com",         # EMAIL_ADDRESS
            "IP interna 192.168.1.1 expuesta",        # IP_ADDRESS
            "Teléfono: +56 9 8765 4321, urgente",     # PHONE_NUMBER
            "Tarjeta 4111-1111-1111-1111 caduca 12/26",  # CREDIT_CARD

            # ── Textos normales (no deben tocarse) ────────────────────────
            "El precio es 10.000 pesos.",
            "Probabilidad: P(A ∩ B) = 0.5",
            "Número normal 3333",
            "Esto no tiene nada sensible",
        ]
    }
    return Dataset.from_dict(data)


# --------------------------------------------------------------------------- #
# Procesamiento por lote                                                      #
# --------------------------------------------------------------------------- #


def process_batch(
    examples: Dict[str, Any],
    *,
    col: str,
    entities: List[str],
    language: str,
    max_len: int,
) -> Dict[str, Any]:
    analyzer, anonymizer = ensure_presidio()
    texts = examples[col]
    new_texts: List[str] = []

    # ------------------------------------------------------------ Presidio --
    valid_mask = [isinstance(t, str) and len(t) <= max_len for t in texts]
    valid_texts = [t for t, ok in zip(texts, valid_mask) if ok]

    if valid_texts:
        if hasattr(analyzer, "analyze_batch"):
            detections = analyzer.analyze_batch(
                text=valid_texts, language=language, entities=entities
            )
        else:
            detections = [analyzer.analyze(t, language=language, entities=entities)
                          for t in valid_texts]
    else:
        detections = []

    idx = 0
    for t, ok in zip(texts, valid_mask):
        if ok:
            pii_clean = anonymizer.anonymize(text=t,
                                             analyzer_results=detections[idx]).text
            idx += 1
        else:
            pii_clean = t  # demasiado largo → no procesar PII

        # --------------------------------------------------- Reemplazo IDs --
        final_text = replace_identifiers(pii_clean)
        new_texts.append(final_text)

    examples[col] = new_texts
    return examples


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anonimiza PII (Presidio) + IDs latinoamericanos en un solo paso."
    )
    parser.add_argument("--input_path", help="Ruta al dataset Hugging Face")
    parser.add_argument("--output_path", required=True,
                        help="Ruta destino (save_to_disk)")
    parser.add_argument("--column", default="texto",
                        help="Columna de texto (default: 'texto')")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_proc", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--max_text_len", type=int, default=100_000)
    parser.add_argument("--entities", 
                        default="EMAIL_ADDRESS,IP_ADDRESS,PHONE_NUMBER,CREDIT_CARD",
                        help="Entidades Presidio separadas por coma")
    parser.add_argument("--language", default="en")
    parser.add_argument("--demo", action="store_true", help="Dataset de demostración")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"

    entities = [e.strip() for e in args.entities.split(",") if e.strip()]
    t0 = time.perf_counter()

    # --------------------------------------------------------------------- #
    # Cargar dataset                                                         #
    # --------------------------------------------------------------------- #
    if args.demo:
        print("[INFO] Usando dataset de demostración (--demo)")
        ds = build_demo_dataset(col=args.column)
    else:
        if not args.input_path:
            raise ValueError("Debe indicar --input_path o usar --demo")
        print("[INFO] Cargando dataset…")
        ds = load_from_disk(args.input_path)

    if args.column not in ds.column_names:
        raise ValueError(f"La columna '{args.column}' no existe en el dataset.")

    print(f"[INFO] Filas originales: {ds.num_rows:,}")

    # --------------------------------------------------------------------- #
    # Procesar en paralelo                                                   #
    # --------------------------------------------------------------------- #
    ds_final = ds.map(
        process_batch,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.num_proc,
        desc="🔐 Presidio + IDs",
        fn_kwargs=dict(
            col=args.column,
            entities=entities,
            language=args.language,
            max_len=args.max_text_len,
        ),
        load_from_cache_file=False,
        writer_batch_size=args.batch_size,
        keep_in_memory=False,
    )

    # --------------------------------------------------------------------- #
    # Guardar                                                                #
    # --------------------------------------------------------------------- #
    print(f"[INFO] Guardando en {args.output_path} …")
    ds_final.save_to_disk(args.output_path)
    print(f"[✓] Terminado en {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
