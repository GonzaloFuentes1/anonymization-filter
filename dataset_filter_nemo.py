import argparse
from time import time

import pandas as pd
from dask import compute, delayed
from datasets import Dataset, load_from_disk
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client
from tqdm import tqdm


def process_batch(batch, modifier):
    """
    Procesa un batch de textos, aplicando anonimización de PII.

    Args:
        batch (dict): Contiene la clave "texto".
        modifier (PiiModifier): Modificador para aplicar anonimización.

    Returns:
        pd.DataFrame: DataFrame con los textos modificados.
    """
    df = pd.DataFrame({"text": batch["texto"]})
    dataset = DocumentDataset.from_pandas(df)
    modified = Modify(modifier)(dataset)
    return modified.to_pandas()


def main():
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Anonimización de textos con NeMo Curator")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Ruta al dataset Hugging Face")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Ruta para guardar el dataset anonimizado")
    parser.add_argument(
        "--supported_entities",
        type=str,
        nargs="+",
        default=[
            "CREDIT_CARD",
            "EMAIL_ADDRESS",
            "IP_ADDRESS",
            "US_SSN",
            "US_PASSPORT",
            "US_DRIVER_LICENSE",
            "PHONE_NUMBER",
        ],
        help="Lista de entidades PII a anonimizar (separadas por espacio)"
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Dispositivo a utilizar para la anonimización (cpu o gpu)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Cantidad de muestras por batch"
    )
    args = parser.parse_args()

    # Inicializar cliente Dask (4 workers con 2 hilos c/u)
    client = get_client(cluster_type=args.device, n_workers=4, threads_per_worker=2)
    print("Dashboard en:", client.dashboard_link)

    # Lista completa de entidades soportadas (para referencia):
    # "ADDRESS", "CREDIT_CARD", "EMAIL_ADDRESS", "DATE_TIME",
    # "IP_ADDRESS", "LOCATION", "PERSON", "URL",
    # "US_SSN", "US_PASSPORT", "US_DRIVER_LICENSE", "PHONE_NUMBER"

    modifier = PiiModifier(
        language="en",
        supported_entities=args.supported_entities,
        anonymize_action="replace",
        batch_size=args.batch_size,
        device=args.device,
    )

    # Cargar dataset
    ds = load_from_disk(args.input_path)
    total_samples = len(ds)

    delayed_results = []
    for start in tqdm(range(0, total_samples, args.batch_size)):
        end = min(start + args.batch_size, total_samples)
        print(f"Procesando muestras {start} a {end}...")
        batch = ds[start:end]
        delayed_results.append(delayed(process_batch)(batch, modifier))

    processed = compute(*delayed_results)
    final_df = pd.concat(processed, ignore_index=True)
    final_df["meta"] = ds["meta"]

    for _, doc in final_df.iterrows():
        print("Texto anonimizado:", doc["text"])
        print("=" * 60)

    Dataset.from_pandas(final_df).save_to_disk(args.output_path)


if __name__ == "__main__":
    t1 = time()
    main()
    print("Tiempo total:", time() - t1)
