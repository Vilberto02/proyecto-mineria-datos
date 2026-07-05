import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import os

def load_and_align_datasets():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(current_dir, '..', 'datos'))
    if not os.path.exists(base_path):
        base_path = os.path.join(current_dir, 'datos') # fallback
    
    # Cargar tiktok
    df_tiktok = pd.read_csv(os.path.join(base_path, 'dataset_procesado_tiktok_parte01.csv'), sep=';')
    # Renombrar columnas para que coincidan con un estándar
    df_tiktok = df_tiktok.rename(columns={
        'Name ': 'source_name', 
        'Comment URL': 'source_url', 
        'Date': 'date_published'
    })
    
    # Cargar youtube 02 (verified)
    df_yt2 = pd.read_csv(os.path.join(base_path, 'dataset_procesado_youtube_parte02.csv'), sep=';')
    # Renombrar si es necesario. Veamos las columnas para yt2 (asumiendo que son iguales a yt1)
    if 'Name ' in df_yt2.columns:
        df_yt2 = df_yt2.rename(columns={'Name ': 'source_name', 'Comment URL': 'source_url', 'Date': 'date_published'})
    
    # Cargar youtube 01 (unverified)
    df_yt1 = pd.read_csv(os.path.join(base_path, 'dataset_procesado_youtube_parte01.csv'), sep=';')
    
    # Agregar una bandera para rastrear el estado de verificación
    df_tiktok['is_verified'] = True
    df_yt2['is_verified'] = True
    df_yt1['is_verified'] = False
    
    # Agregar bandera de origen
    df_tiktok['origin'] = 'tiktok'
    df_yt2['origin'] = 'youtube_02'
    df_yt1['origin'] = 'youtube_01'
    
    # Asegurar que todos los nombres de columna estén limpios (eliminar comillas, espacios en blanco y \r)
    for df in [df_tiktok, df_yt2, df_yt1]:
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace('\r', '')
            
    # Limpiar etiquetas de emoción (eliminar espacios en blanco al inicio/final)
    df_tiktok['emocion'] = df_tiktok['emocion'].str.strip()
    df_yt2['emocion'] = df_yt2['emocion'].str.strip()
    df_yt1['emocion'] = df_yt1['emocion'].str.strip()
    
    # Keep common columns
    common_cols = ['source_name', 'source_url', 'content', 'date_published', 'emocion', 'content_global', 'tokens', 'lemas', 'content_clasico', 'content_transformers', 'is_verified', 'origin']
    
    df_tiktok = df_tiktok[[c for c in common_cols if c in df_tiktok.columns]]
    df_yt2 = df_yt2[[c for c in common_cols if c in df_yt2.columns]]
    df_yt1 = df_yt1[[c for c in common_cols if c in df_yt1.columns]]
    
    return df_tiktok, df_yt2, df_yt1

def main():
    print("Cargando datasets...")
    df_tiktok, df_yt2, df_yt1 = load_and_align_datasets()
    
    print(f"Filas verificadas de TikTok: {len(df_tiktok)}")
    print(f"Filas verificadas de YouTube 02: {len(df_yt2)}")
    print(f"Filas no verificadas de YouTube 01: {len(df_yt1)}")
    
    # Combina los datasets verificados
    df_verified = pd.concat([df_tiktok, df_yt2], ignore_index=True)
    
    # Queremos exactamente 1000 muestras para el Gold Set mediante muestreo estratificado
    gold_set_size = 1000
    
    # Filtra las filas con NaN en 'emocion' por si acaso
    df_verified = df_verified.dropna(subset=['emocion'])
    
    # Realiza el muestreo estratificado
    # Como solo queremos 1000, el test_size es 1000 / len(df_verified)
    split = StratifiedShuffleSplit(n_splits=1, test_size=gold_set_size, random_state=42)
    
    for train_index, test_index in split.split(df_verified, df_verified['emocion']):
        # test_index contains the 1000 samples
        gold_set = df_verified.iloc[test_index].copy()
        verified_remaining = df_verified.iloc[train_index].copy()
        
    print(f"\nGold Set creado con {len(gold_set)} filas.")
    print("Distribución de emociones de Gold Set:")
    print(gold_set['emocion'].value_counts())
    
    # El dataset Silver Target es el conjunto verificado restante + todos los datos no verificados
    silver_target_dataset = pd.concat([verified_remaining, df_yt1], ignore_index=True)
    print(f"\nSilver Target Dataset creado con {len(silver_target_dataset)} filas.")
    
    # Guardar
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(current_dir, '..', 'datos'))
    if not os.path.exists(output_dir):
        output_dir = os.path.join(current_dir, 'datos')
    
    gold_set.to_csv(os.path.join(output_dir, 'gold_set.csv'), sep=';', index=False)
    silver_target_dataset.to_csv(os.path.join(output_dir, 'silver_target_dataset.csv'), sep=';', index=False)
    print("Datasets guardados correctamente.")

if __name__ == "__main__":
    main()
