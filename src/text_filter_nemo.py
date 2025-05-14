import pandas as pd
from time import time
from math import ceil
from tqdm import tqdm
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client


def process_chunk(text_chunk, supported_entities, batch_size, device="cpu"):
    df = pd.DataFrame({"text": text_chunk})
    df["original_text"] = df["text"]

    doc_dataset = DocumentDataset.from_pandas(df)
    modifier = PiiModifier(
        language="en",
        supported_entities=supported_entities,
        anonymize_action="replace",
        batch_size=batch_size,
        device=device,
    )
    modified = Modify(modifier)(doc_dataset)
    return modified.to_pandas()


def anonymize_large_list_parallel(text_list, chunk_size=10000, n_workers=16, threads_per_worker=1):
    client = get_client(cluster_type="cpu", n_workers=n_workers, threads_per_worker=threads_per_worker)
    print("✅ Dask dashboard:", client.dashboard_link)

    supported_entities = [
        "CREDIT_CARD", "EMAIL_ADDRESS", "IP_ADDRESS", "PHONE_NUMBER"
    ]

    # Dividir en chunks
    total_chunks = ceil(len(text_list) / chunk_size)
    chunks = [text_list[i * chunk_size:(i + 1) * chunk_size] for i in range(total_chunks)]

    print(f"🚀 Procesando {len(text_list)} textos en {total_chunks} chunks de {chunk_size}...")

    # Enviar tareas paralelas
    futures = []
    for chunk in chunks:
        fut = client.submit(
            process_chunk,
            chunk,
            supported_entities,
            batch_size=512,
            device="cpu"
        )
        futures.append(fut)

    # Recolectar resultados con barra de progreso
    results = []
    for fut in tqdm(futures, desc="Procesando chunks"):
        results.append(fut.result())

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    # Simulación con 40k textos
    textos_prueba = [
        "Mi correo es juan.perez@gmail.com y mi número es 987654321.",
        "Número de tarjeta: 4111-1111-1111-1111, IP: 192.168.0.1",
        "SSN: 123-45-6789, Teléfono: (555) 123-4567",
        "Hoy es 2023-05-01 y compré por 20.000 CLP.",
        "Nada que anonimizar aquí."
    ] * 1200000

    t1 = time()
    df_result = anonymize_large_list_parallel(textos_prueba, chunk_size=1000, n_workers=64)
    t2 = time()

    print(df_result.head(5))
    print(f"\n✅ Tiempo total: {t2 - t1:.2f} segundos.")
