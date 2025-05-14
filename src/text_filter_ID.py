import re
import time

# Expresiones regulares para identificadores personales
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
            etiqueta = "<ID>"
            reemplazo = list(etiqueta.ljust(end - start))
            resultado[start:end] = reemplazo[:end - start]
            for i in range(start, end):
                ocupado[i] = True

    return "".join(resultado)


# Lista de ejemplos
textos = [
    "Mi RUT es 12.345.678-9.",
    "CUIT argentino: 20-12345678-1.",
    "CURP mexicana: ABCD901231HDFRRA01.",
    "RFC: MARA900101ABC",
    "CI de Uruguay: 1.234.567-8.",
    "RTN de Honduras: 0801-1989-12345.",
    "DPI guatemalteco: 1234 56789 0101.",
    "ID de Haití: 12-01-99-12345.",
    "Nací en 1995 y tengo 2 hermanos.",
    "Mi teléfono es 987654321, no es un ID.",
    "El precio fue de 35000 CLP.",
    "Vivo en el departamento 404.",
    "El número de pedido es 2023-04.",
    "1", "11", "111", "1111", "11111",
    "Tengo 8 años de experiencia.",
    "En 2020 viajé a México.",
]

# Evaluación con pausa entre cada uno
for texto in textos:
    print("Original:    ", texto)
    print("Anonimizado: ", reemplazar_identificadores(texto))
    print("-" * 50)
    time.sleep(0.5)
