import pandas as pd
from time import time
from math import ceil
from tqdm import tqdm
import random
import json
import os
from nemo_curator.datasets import DocumentDataset
from nemo_curator.modifiers.pii_modifier import PiiModifier
from nemo_curator.modules.modify import Modify
from nemo_curator.utils.distributed_utils import get_client
from dask.distributed import as_completed
import matplotlib.pyplot as plt


def crear_texto_simple(categoria):
    """Crea texto simple con información PII según categoría"""
    if categoria == "corto":
        # ~100 caracteres con un dato PII
        relleno = "a" * random.randint(30, 70)
        dato_pii = f"correo: user{random.randint(1,999)}@example.com"
        return f"{relleno} {dato_pii}"
        
    elif categoria == "medio":
        # ~500 caracteres con dos datos PII
        relleno = "b" * random.randint(400, 450)
        dato_pii1 = f"correo: user{random.randint(1,999)}@example.com"
        dato_pii2 = f"teléfono: +1-555-{random.randint(100,999)}-{random.randint(1000,9999)}"
        return f"{relleno[:200]} {dato_pii1} {relleno[200:350]} {dato_pii2} {relleno[350:]}"
        
    else:  # largo
        # ~2000 caracteres con tres datos PII
        relleno = "c" * random.randint(1800, 1900)
        dato_pii1 = f"correo: user{random.randint(1,999)}@example.com"
        dato_pii2 = f"teléfono: +1-555-{random.randint(100,999)}-{random.randint(1000,9999)}"
        dato_pii3 = f"tarjeta: {random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
        return f"{relleno[:600]} {dato_pii1} {relleno[600:1200]} {dato_pii2} {relleno[1200:1700]} {dato_pii3} {relleno[1700:]}"


def generar_conjuntos_prueba(n_textos=3000000):
    """Genera tres conjuntos de prueba simples"""
    print(f"Generando {n_textos} textos por categoría...")
    
    conjuntos = {
        "corto": [],
        "medio": [],
        "largo": []
    }
    
    for categoria in conjuntos.keys():
        for _ in tqdm(range(n_textos), desc=f"Generando textos {categoria}s"):
            conjuntos[categoria].append(crear_texto_simple(categoria))
    
    return conjuntos


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


def anonymize_large_list_parallel(text_list, chunk_size, n_workers, threads_per_worker, batch_size=256):
    """Procesa una lista grande de textos en paralelo usando Dask"""
    client = get_client(
        cluster_type="cpu", 
        n_workers=n_workers, 
        threads_per_worker=threads_per_worker
    )
    print(f"✅ Dask dashboard: {client.dashboard_link}")
    print(f"✅ Configuración: {n_workers} workers, {threads_per_worker} threads por worker")

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
            batch_size=batch_size,
            device="cpu"
        )
        futures.append(fut)

    # Recolectar resultados con barra de progreso
    results = []
    with tqdm(total=len(futures), desc="Procesando chunks") as pbar:
        for completed in as_completed(futures):
            df_chunk = completed.result()
            results.append(df_chunk)
            pbar.update(1)

    # Cerrar cliente Dask
    client.close()

    return pd.concat(results, ignore_index=True)


def ejecutar_benchmark(conjuntos, chunk_sizes, worker_thread_configs):
    """Ejecuta benchmarks con diferentes configuraciones de workers y threads"""
    resultados = []
    
    # Registrar longitudes promedio de cada conjunto
    longitudes = {
        tipo: sum(len(texto) for texto in textos[:100]) / 100
        for tipo, textos in conjuntos.items()
    }
    
    print("Longitudes promedio de los conjuntos:")
    for tipo, longitud in longitudes.items():
        print(f"- {tipo}: {longitud:.1f} caracteres")
    
    # Guardar la configuración del experimento
    os.makedirs("resultados_benchmark", exist_ok=True)
    
    # Para cada tipo de texto
    for tipo, textos in conjuntos.items():
        print(f"\n\n{'='*70}")
        print(f"BENCHMARKS PARA TEXTOS {tipo.upper()}")
        print(f"{'='*70}")
        
        textos_prueba = textos
        
        # Para cada configuración de workers y threads
        for n_workers, threads_per_worker in worker_thread_configs:
            print(f"\n{'='*50}")
            print(f"CONFIGURACIÓN: {n_workers} workers, {threads_per_worker} threads/worker")
            print(f"{'='*50}")
            
            # Para cada tamaño de chunk
            for chunk_size in chunk_sizes:
                print(f"\n{'-'*40}")
                print(f"Chunk size: {chunk_size}")
                print(f"{'-'*40}")
                
                # Ejecutar la prueba
                t1 = time()
                df_result = anonymize_large_list_parallel(
                    textos_prueba, 
                    chunk_size=chunk_size,
                    n_workers=n_workers,
                    threads_per_worker=threads_per_worker
                )
                t2 = time()
                
                tiempo_total = t2 - t1
                textos_por_segundo = len(textos_prueba) / tiempo_total
                
                print(f"✅ Tiempo total: {tiempo_total:.2f} segundos")
                print(f"✅ Rendimiento: {textos_por_segundo:.2f} textos/segundo")
                
                # Guardar resultado
                resultado = {
                    "tipo_texto": tipo,
                    "n_workers": n_workers,
                    "threads_per_worker": threads_per_worker,
                    "chunk_size": chunk_size,
                    "n_textos": len(textos_prueba),
                    "tiempo_total": tiempo_total,
                    "textos_por_segundo": textos_por_segundo
                }
                
                resultados.append(resultado)
                
                # Guardar resultados parciales inmediatamente
                df_resultados = pd.DataFrame(resultados)
                df_resultados.to_csv("resultados_benchmark/benchmark_workers_threads.csv", index=False)
    
    return resultados


def visualizar_resultados(csv_path="resultados_benchmark/benchmark_workers_threads.csv"):
    """Genera visualizaciones a partir de los resultados"""
    if not os.path.exists(csv_path):
        print(f"⚠️ No se encontró el archivo {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # 1. Gráfico de barras por configuración de workers/threads
    plt.figure(figsize=(15, 10))
    
    # Agrupar por tipo de texto y configuración de workers/threads
    for tipo in df['tipo_texto'].unique():
        subset = df[df['tipo_texto'] == tipo]
        configs = subset.apply(lambda row: f"{row['n_workers']}w/{row['threads_per_worker']}t", axis=1).unique()
        
        medias = []
        etiquetas = []
        
        for config in configs:
            n_workers, threads = config.split('w/')[0], config.split('w/')[1][:-1]
            datos = subset[(subset['n_workers'] == int(n_workers)) & 
                          (subset['threads_per_worker'] == int(threads))]
            media = datos['textos_por_segundo'].mean()
            medias.append(media)
            etiquetas.append(config)
        
        plt.figure(figsize=(12, 6))
        plt.bar(etiquetas, medias, label=f'Textos {tipo}')
        plt.title(f'Rendimiento promedio por configuración - Textos {tipo}')
        plt.xlabel('Configuración (workers/threads)')
        plt.ylabel('Textos/segundo')
        plt.grid(axis='y')
        plt.tight_layout()
        plt.savefig(f"resultados_benchmark/rendimiento_{tipo}.png")
    
    # 2. Gráfico comparativo por chunk size
    plt.figure(figsize=(15, 10))
    
    for tipo in df['tipo_texto'].unique():
        subset = df[df['tipo_texto'] == tipo]
        chunk_sizes = subset['chunk_size'].unique()
        
        plt.figure(figsize=(12, 6))
        for chunk_size in chunk_sizes:
            datos = subset[subset['chunk_size'] == chunk_size]
            configs = datos.apply(lambda row: f"{row['n_workers']}w/{row['threads_per_worker']}t", axis=1)
            plt.plot(configs, datos['textos_por_segundo'], 'o-', label=f'Chunk {chunk_size}')
        
        plt.title(f'Rendimiento por tamaño de chunk - Textos {tipo}')
        plt.xlabel('Configuración (workers/threads)')
        plt.ylabel('Textos/segundo')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"resultados_benchmark/rendimiento_chunks_{tipo}.png")
    
    # 3. Tabla de resumen
    mejores = df.loc[df.groupby('tipo_texto')['textos_por_segundo'].idxmax()]
    mejores.to_csv("resultados_benchmark/mejores_configuraciones.csv", index=False)
    
    print("✅ Visualizaciones generadas en carpeta 'resultados_benchmark'")


if __name__ == "__main__":
    # Configuración del experimento - usando todos los núcleos disponibles
    NUM_TEXTOS = 10000000  # 10 millones de textos por categoría
    
    # Chunks más grandes como solicitaste
    CHUNK_SIZES = [100000, 50000, 10000]
    
    # Configuraciones que aprovechan todos los núcleos disponibles (128)
    WORKER_THREAD_CONFIGS = [
        (60,1),
        (50,1),
        (40,1),
        (30,1),
        (20,1),
        (10,1),
    ]
    
    # Generar o cargar conjuntos de prueba
    archivo_conjuntos = "textos_prueba_simple_10M.json"
    if os.path.exists(archivo_conjuntos):
        print(f"Cargando conjuntos desde {archivo_conjuntos}...")
        with open(archivo_conjuntos, "r") as f:
            conjuntos = json.load(f)
    else:
        conjuntos = generar_conjuntos_prueba(NUM_TEXTOS)
        print(f"Guardando conjuntos en {archivo_conjuntos}...")
        with open(archivo_conjuntos, "w") as f:
            json.dump(conjuntos, f)
    
    # Ejecutar benchmarks
    resultados = ejecutar_benchmark(
        conjuntos,
        CHUNK_SIZES,
        WORKER_THREAD_CONFIGS
    )
    
    # Visualizar resultados
    visualizar_resultados()
    
    print("\n✅ Experimento completo! Resultados en 'resultados_benchmark/'")