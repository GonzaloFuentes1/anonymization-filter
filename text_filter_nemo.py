import pandas as pd
from datasets import Dataset
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client


def anonymize_text_list(text_list, batch_size=1000, save_dataset=False,
                        output_path=None):
    """
    Aplica anonimización de PII sobre una lista de textos.

    Args:
        text_list (list): Lista de strings a anonimizar
        batch_size (int): Tamaño de lote para procesamiento
        save_dataset (bool): Si se debe guardar el dataset anonimizado
        output_path (str): Carpeta donde guardar el dataset si save_dataset es True
    """
    if not text_list:
        raise ValueError("La lista de texto está vacía")

    client = get_client(cluster_type="cpu", n_workers=2, threads_per_worker=1)
    print("Dashboard en:", client.dashboard_link)

    modifier = PiiModifier(
        language="en",
        supported_entities=[
            "CREDIT_CARD",
            "EMAIL_ADDRESS",
            "IP_ADDRESS",
            "US_SSN",
            "PHONE_NUMBER",
        ],
        anonymize_action="replace",
        batch_size=batch_size,
        device="cpu",
    )

    df = pd.DataFrame({"text": text_list})
    df["original_text"] = df["text"]

    dataset = DocumentDataset.from_pandas(df)
    modified_dataset = Modify(modifier)(dataset)
    df_result = modified_dataset.to_pandas()

    for _, row in df_result.iterrows():
        print("Original:    ", row["original_text"])
        print("Anonimizado: ", row["text"])
        print("-" * 60)

    if save_dataset and output_path:
        Dataset.from_pandas(df_result).save_to_disk(output_path)
        print(f"Dataset guardado en: {output_path}")

    return df_result


textos_prueba = [
    "Mi correo es juan.perez@gmail.com y mi número es 987654321.",
    "Número de tarjeta: 4111-1111-1111-1111, IP: 192.168.0.1",
    "SSN: 123-45-6789, Teléfono: (555) 123-4567",
    "Hoy es 2023-05-01 y compré por 20.000 CLP.",
    "Nada que anonimizar aquí."
]

anonymize_text_list(textos_prueba)
