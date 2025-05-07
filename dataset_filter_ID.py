import argparse
import re

from datasets import Dataset, load_from_disk
from tqdm import tqdm

# === Expresiones regulares por país o tipo de identificador ===
ordered_regex_patterns = {
    "NIT_SLV": r"\b\d{4}(?:[-\s])\d{6}(?:[-\s])\d{3}(?:[-\s])\d\b",
    "RNC_DOM": r"\b\d{1}(?:[-\s])\d{2}(?:[-\s])\d{5}(?:[-\s])\d\b",
    "RTN_HND": r"\b\d{4}(?:[-\s])\d{4}(?:[-\s])\d{5}\b",
    "CI_NIC": r"\b\d{3}(?:[-\s])\d{6}(?:[-\s])\d{4}[A-Z]?\b",
    "RUC_NIC": r"\b\d{3}(?:[-\s])\d{6}(?:[-\s])\d{4}[A-Z]?\b",
    "RUC_ECU": r"\b\d{10}(?:[-\s])\d{3}\b",
    "CIE_DOM": r"\b\d{3}(?:[-\s])\d{7}(?:[-\s])\d\b",
    "CPF_BRA": r"\b\d{3}(?:[.\s-])\d{3}(?:[.\s-])\d{3}(?:[.\s-])\d{2}\b",
    "RUC_PAN": r"\b\d{1,4}(?:[-\s])\d{1,4}(?:[-\s])\d{1,4}\b",
    "CI_CUB": r"\b\d{6}(?:[-\s])\d{5}\b",
    "CI_CRI": r"\b\d{1}(?:[-\s])\d{4}(?:[-\s])\d{4}\b",
    "DPI_GTM": r"\b\d{4}\s\d{5}\s\d{4}\b",
    "CURP_MEX": r"\b[A-Z]{4}[-]?\d{6}[-]?[HM][A-Z]{5}[A-Z0-9]\b",
    "RFC_MEX": r"\b[A-ZÑ&]{3,4}[-]?\d{6}[-]?[A-Z0-9]{3}\b",
    "RIF_VEN": r"\b[JGVEP](?:[-\s])\d{8}(?:[-\s])\d\b",
    "CI_BOL": r"\b\d{6,8}(?:[-\s])[A-Z]{2}\b",
    "RUC_PRY": r"\b\d{6,8}[A-Z]?(?:[-\s])\d\b",
    "CUIT_ARG": r"\b\d{2}[.\s-]\d{8}[.\s-]\d\b",
    "NIT_COL": r"\b\d{8,10}-\d\b",
    "CI_URU": r"\b\d{1,2}[.\s-]\d{3}[.\s-]\d{3}[.\s-]\d\b",
    "RUT_CHI": r"\b\d{1,2}[.\s-]\d{3}[.\s-]\d{3}[.\s-]?[\dkK]\b",
    "CI_VEN": r"\b[VvEe](?:[-\s])\d{6,8}\b",
    "PAS_ARG": r"\bAA(?:[-\s])\d{7}\b",
    "PAS_CHI": r"\b[Cc](?:[-\s])\d{8}\b",
    "PAS_MEX": r"\bG(?:[-\s])\d{8}\b",
    "NIT_GTM": r"\b\d{6,8}(?:[-\s])\d\b",
    "NIT_BOL": r"\b\d{6,8}(?:[-\s])\d\b",
    "ID_HTI": r"\b\d{2}(?:[-\s])\d{2}(?:[-\s])\d{2}(?:[-\s])\d{5}\b"
}


def reemplazar_identificadores(texto):
    """
    Reemplaza cualquier identificador detectado con la etiqueta <ID>
    respetando solapamientos.
    """
    reemplazos = []

    for _, patron in ordered_regex_patterns.items():
        for match in re.finditer(patron, texto, flags=re.IGNORECASE):
            start, end = match.span()
            reemplazos.append((start, end))

    reemplazos.sort(key=lambda x: (- (x[1] - x[0]), x[0]))
    ocupado = [False] * len(texto)
    resultado = list(texto)

    for start, end in reemplazos:
        if not any(ocupado[start:end]):
            etiqueta_texto = "<ID>"
            reemplazo = list(etiqueta_texto.ljust(end - start))
            resultado[start:end] = reemplazo[:end - start]
            for i in range(start, end):
                ocupado[i] = True

    return "".join(resultado)


def main():
    parser = argparse.ArgumentParser(
        description="Anonimiza identificadores en texto con <ID>")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Ruta al dataset Hugging Face")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Ruta para guardar el dataset anonimizado")
    parser.add_argument("--column", type=str, default="text",
                        help="Nombre de la columna de texto (default: 'text')")
    args = parser.parse_args()

    print("[INFO] Cargando dataset...")
    ds = load_from_disk(args.input_path)

    if args.column not in ds.column_names:
        raise ValueError(
            f"La columna '{args.column}' no existe en el dataset")

    print("[INFO] Procesando textos...")
    textos_modificados = []
    for texto in tqdm(ds[args.column], desc="Anonimizando"):
        textos_modificados.append(reemplazar_identificadores(texto))

    print("[INFO] Generando dataset final...")
    new_ds = Dataset.from_dict({args.column: textos_modificados})

    print(f"[INFO] Guardando en {args.output_path}...")
    new_ds.save_to_disk(args.output_path)
    print("[✓] Proceso completado.")


if __name__ == "__main__":
    main()
