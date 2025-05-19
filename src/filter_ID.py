#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Anonimizador de identificadores latinoamericanos

• Reemplaza cualquier ID reconocido por la etiqueta <ID>,
manteniendo la longitud del texto para no desalinear offsets.
• Permite trabajar con datasets Hugging Face o un dataset de demostración.

Licencia: CC BY-NC-ND 4.0
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, List, Tuple

from datasets import Dataset, load_from_disk

# --------------------------------------------------------------------------- #
# Expresiones regulares por país / tipo                                        #
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

# Compilamos una sola vez para mayor velocidad (IGNORECASE en todos)
PATTERNS: List[re.Pattern[str]] = [
    re.compile(p, flags=re.IGNORECASE) for p in RAW_PATTERNS.values()
]

# --------------------------------------------------------------------------- #
# Utilidades                                                                  #
# --------------------------------------------------------------------------- #


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


def build_demo_dataset(col: str = "text") -> Dataset:
    """Devuelve un dataset de demostración con casos reales y falsos positivos."""
    data = {
        col: [
            # ─── Identificadores reales ─────────────────────────────────────────
            "Mi RUT es 12.345.678-9",                          # CHI
            "CURP: GOML840512HDFRRN09",                        # MEX
            "RFC: GOM8405121A1",                               # MEX
            "Número cubano: CUB-123456-54321",                 # CUB
            "DPI guatemalteco: 1234 56789 1234",               # GTM
            "ID Haití: 01-02-03-12345",                        # HTI
            "CI Bolivia: 12345678-LP",                         # BOL
            "CUIT: 20-12345678-1",                             # ARG
            "Número brasileño: 123.456.789-00",                # BRA
            "Cédula Venezuela: V-12345678",                    # VEN
            "CI uruguaya: 1.234.567-8",                        # URU
            "Pasaporte chileno: C-12345678",                   # CHI
            "Pasaporte mexicano: G-12345678",                  # MEX
            "Pasaporte argentino: AA-1234567",                 # ARG
            "Número salvadoreño: SV-1234-123456-123-1",        # SLV
            "Número dominicano: RD-1-23-12345-6",              # DOM
            "RTN Honduras: HN-1234-5678-12345",                # HND
            "RUC Panamá: P-123-456-789",                       # PAN
            "RUC Perú: PE-20123456789",                        # PER
            "RUC Paraguay: 12345678A-9",                       # PRY
            "RUC Ecuador: EC-1790012345-001",                  # ECU
            "CI Nicaragua: 123-456789-1234A",                  # NIC
            "Número colombiano: COL-800123456-1",              # COL

            # ─── Frases con números normales (no deberían sustituirse) ─────────
            "El precio es 10.000 pesos.",
            "Hoy es 12-05-2025, temperatura 22.5°C.",
            "La ecuación es: 5x² + 3x - 7 = 0",
            "Mi número de serie es 123456789",
            "La raíz de 64 es 8",
            "Esto no tiene nada",
            "Código de producto: A1B2C3D4E5",
            "Resultado de 2023/6 = 337.17",
            "Probabilidad: P(A ∩ B) = 0.5",
            "Suma de 7 + 8 + 9 = 24",

            # ─── Casos diseñados para falsos positivos ──────────────────────────
            "Transacción: 123-456-789",
            "Factura número 1234-5678",
            "Fecha de emisión: 10-10-2022",
            "Código interno: 01-02-3456",
            "Folio: 987654321-0",
            "Serial: 123456-789",
            "12-12-1212 es una fecha espeluznante",
            "Nota de débito 12345678-9",
            "Token ID: 1122-3344-5566",
            "Rango: 100-200-300",
            "Número cliente: 1234567",
            "Clave: 0102030405",
            "Código SII: 12.345.678-9",
            "Monto: 12000-2000",
            "Error 202-404-500",
            "Teléfono: 1234-5678",
            "Referencia: 1111-2222-3333",
            "Documento: 2023-05-17",
            "Identificador: 20-12345678-1",
            "Mi IP es 192.168.1.1",
            "Numero normal 1",
            "Numero normal 11",
            "Numero normal 222",
            "Numero normal 3333",
            "Numero normal 44444",
            "Numero normal 555555",
            "Numero normal 6666666",
            "Numero normal 77777777",
            "Numero normal 888888888",
            "Numero normal 9999999999",
            "Número de teléfono: +56 9 8765 4321",
        ]
    }
    return Dataset.from_dict(data)


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Anonimiza identificadores personales con la etiqueta <ID>."
    )
    parser.add_argument("--input_path", help="Ruta al dataset Hugging Face")
    parser.add_argument("--output_path", required=True,
                        help="Ruta para guardar el dataset anonimizado")
    parser.add_argument("--column", default="text",
                        help="Nombre de la columna de texto (default: 'text')")
    parser.add_argument("--demo", action="store_true",
                        help="Usa un dataset de demostración")
    args = parser.parse_args()

    # --------------------------------------------------------------------- #
    # Cargar dataset                                                         #
    # --------------------------------------------------------------------- #
    if args.demo:
        print("[INFO] Usando dataset de demostración (--demo)")
        ds = build_demo_dataset(args.column)
    else:
        if not args.input_path:
            raise ValueError("Debe especificar --input_path o usar --demo")
        print("[INFO] Cargando dataset desde disco…")
        ds = load_from_disk(args.input_path)

    if args.column not in ds.column_names:
        raise ValueError(f"La columna '{args.column}' no existe en el dataset.")

    # --------------------------------------------------------------------- #
    # Anonimizar                                                             #
    # --------------------------------------------------------------------- #
    print("[INFO] Anonimizando identificadores…")
    ds_anonymized = ds.map(
        lambda ex: {args.column: replace_identifiers(ex[args.column])},
        desc="🔐 Reemplazando",
    )

    # --------------------------------------------------------------------- #
    # Guardar                                                               #
    # --------------------------------------------------------------------- #
    print(f"[INFO] Guardando resultado en {args.output_path}…")
    ds_anonymized.save_to_disk(args.output_path)
    print("[✓] Proceso completado.")


if __name__ == "__main__":
    main()
