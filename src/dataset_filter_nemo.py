import argparse
from math import ceil
from time import time
import pandas as pd
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client


def parse_args():
    parser = argparse.ArgumentParser(description="Anonimiza un dataset local de HuggingFace con nemo-curator.")
    parser.add_argument("--input_path", type=str, required=True, help="Ruta al dataset local de HuggingFace.")
    parser.add_argument("--output_path", type=str, required=True, help="Ruta donde guardar el dataset anonimizado.")
    parser.add_argument("--text_column", type=str, default="text", help="Nombre de la columna de texto.")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Tamaño de cada chunk.")
    parser.add_argument("--n_workers", type=int, default=16, help="Número de workers para Dask.")
    parser.add_argument("--threads_per_worker", type=int, default=1, help="Hilos por worker.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Dispositivo de procesamiento.")
    parser.add_argument("--supported_entities", type=str, default="CREDIT_CARD,EMAIL_ADDRESS,IP_ADDRESS,PHONE_NUMBER",
                        help="Entidades a anonimizar (separadas por coma).")
    return parser.parse_args()


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


from dask import delayed, compute

def anonymize_dataset(dataset, text_column, chunk_size, supported_entities, device, n_workers, threads_per_worker):
    from nemo_curator.utils.distributed_utils import get_client
    client = get_client(cluster_type="cpu", n_workers=n_workers, threads_per_worker=threads_per_worker)
    print("✅ Dask dashboard:", client.dashboard_link)

    text_list = dataset[text_column]
    total_chunks = ceil(len(text_list) / chunk_size)
    print(f"🚀 Procesando {len(text_list)} textos en {total_chunks} chunks de {chunk_size}...")

    chunks = [text_list[i * chunk_size:(i + 1) * chunk_size] for i in range(total_chunks)]

    # Usamos dask.delayed para construir tareas sin enviar datos inmediatamente
    delayed_tasks = []
    for chunk in chunks:
        task = delayed(process_chunk)(chunk, supported_entities, batch_size=512, device=device)
        delayed_tasks.append(task)

    # Computar en paralelo
    results = compute(*delayed_tasks, scheduler="distributed", traverse=False)

    # Concatenar resultados
    anonymized_df = pd.concat(results, ignore_index=True)
    dataset = dataset.remove_columns(text_column)
    dataset = dataset.add_column(text_column, anonymized_df["text"].tolist())
    return dataset



def main():
    args = parse_args()
    supported_entities = [e.strip() for e in args.supported_entities.split(",")]

    print("📥 Cargando dataset desde disco...")
    dataset = load_from_disk(args.input_path)

    if isinstance(dataset, DatasetDict):
        result = DatasetDict()
        for split in dataset:
            print(f"🔄 Anonimizando split: {split}")
            t1 = time()
            result[split] = anonymize_dataset(
                dataset[split],
                args.text_column,
                args.chunk_size,
                supported_entities,
                args.device,
                args.n_workers,
                args.threads_per_worker
            )
            print(f"✅ {split} procesado en {time() - t1:.2f} segundos.")
    else:
        print("🔄 Anonimizando dataset...")
        t1 = time()
        result = anonymize_dataset(
            dataset,
            args.text_column,
            args.chunk_size,
            supported_entities,
            args.device,
            args.n_workers,
            args.threads_per_worker
        )
        print(f"✅ Procesado en {time() - t1:.2f} segundos.")

    print("💾 Guardando dataset anonimizado...")
    result.save_to_disk(args.output_path)
    print(f"✅ Guardado en: {args.output_path}")


if __name__ == "__main__":
    main()
