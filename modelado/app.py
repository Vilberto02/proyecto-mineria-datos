import os
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de carpetas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATOS_DIR = os.path.join(BASE_DIR, 'datos')
RESULTADOS_DIR = os.path.join(BASE_DIR, 'resultados')

# Crear carpeta de resultados si no existe
os.makedirs(RESULTADOS_DIR, exist_ok=True)

def cargar_datasets():
    datasets = [
        {'file': 'dataset_tiktok_01_procesado.xlsx', 'origen': 'tiktok_01'},
        {'file': 'dataset_youtube_01_procesado.xlsx', 'origen': 'youtube_01'},
        {'file': 'dataset_youtube_02_procesado.xlsx', 'origen': 'youtube_02'}
    ]
    
    dfs = []
    for ds in datasets:
        file_path = os.path.join(DATOS_DIR, ds['file'])
        if os.path.exists(file_path):
            print(f"   Cargando {ds['file']}...")
            df = pd.read_excel(file_path)
            
            # Limpiar nombres de columnas (eliminar espacios)
            df.columns = df.columns.str.strip().str.lower()
            
            # Verificar que exista la columna emocion
            if 'emocion' not in df.columns:
                print(f"No se encontró la columna 'emocion' en {ds['file']}. Columnas: {df.columns}")
                
            df['origen'] = ds['origen']
            dfs.append(df)
        else:
            print(f"No se encontró {file_path}")
            
    if not dfs:
        raise ValueError("No se encontraron datasets para cargar.")
    
    # Concatenar los datasets
    df_final = pd.concat(dfs, ignore_index=True)
    
    # Limpieza de la columna 'emocion'
    if 'emocion' in df_final.columns:
        df_final['emocion'] = df_final['emocion'].astype(str).str.strip()
        # Estandarizar 'Alegria' a 'Alegría' (considerando posibles problemas de codificación previos)
        df_final['emocion'] = df_final['emocion'].replace({'Alegria': 'Alegría'})
        
        # Validar las emociones según Ekman
        emociones_validas = ['Tristeza', 'Alegría', 'Sorpresa', 'Miedo', 'Asco', 'Ira']
        df_final = df_final[df_final['emocion'].isin(emociones_validas)].copy()
        
    return df_final

def visualizar_datos(df):
    print("\nResumen del dataset combinado:")
    print(f"Total de registros: {len(df)}")
    if 'origen' in df.columns:
        print("\nRegistros por origen:")
        print(df['origen'].value_counts())
    
    if 'emocion' in df.columns:
        print("\nDistribución de emociones:")
        print(df['emocion'].value_counts())
        
        # Estilo de seaborn
        sns.set_theme(style="whitegrid")
        
        # Distribución general de emociones
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='emocion', order=df['emocion'].value_counts().index, palette='viridis')
        plt.title('Distribución General de Emociones')
        plt.xlabel('Emoción')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=45)
        plt.tight_layout()
        ruta_grafico1 = os.path.join(RESULTADOS_DIR, 'distribucion_emociones.png')
        plt.savefig(ruta_grafico1, dpi=300)
        print(f"\nGráfico guardado en: {ruta_grafico1}")
        plt.close()
        
        # Distribución de emociones por origen
        plt.figure(figsize=(12, 6))
        sns.countplot(data=df, x='emocion', hue='origen', order=df['emocion'].value_counts().index, palette='Set2')
        plt.title('Distribución de Emociones por Origen de Datos')
        plt.xlabel('Emoción')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=45)
        plt.legend(title='Origen', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        ruta_grafico2 = os.path.join(RESULTADOS_DIR, 'distribucion_emociones_por_origen.png')
        plt.savefig(ruta_grafico2, dpi=300)
        print(f"Gráfico guardado en: {ruta_grafico2}")
        plt.close()
    else:
        print("No se pudo generar las gráficas por la falta de la columna 'emocion'.")

if __name__ == '__main__':
    print("Iniciando análisis y visualización de datos...")
    try:
        df = cargar_datasets()
        visualizar_datos(df)
        print("Proceso de visualización completado con éxito.")
    except Exception as e:
        print(f"Ocurrió un error: {e}")
