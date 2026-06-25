import os
import sys
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE

# Configuración de carpetas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Apunta a 'modelado'
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

PROYECTO_DIR = os.path.dirname(BASE_DIR) # Apunta a la raíz del proyecto
RESULTADOS_DIR = os.path.join(BASE_DIR, 'resultados')
PREPROC_DIR = os.path.join(PROYECTO_DIR, 'preprocesamiento')

os.makedirs(RESULTADOS_DIR, exist_ok=True)

def cargar_datasets_csv():
    rutas = [
        os.path.join(PREPROC_DIR, 'scrapping-youtube-g4', 'procesado', 'comentarios_clasificado_parte01_clasico.csv'),
        os.path.join(PREPROC_DIR, 'scrapping-youtube-g4', 'procesado', 'comentarios_clasificado_parte02_clasico.csv'),
        os.path.join(PREPROC_DIR, 'scrapping-tiktok-g4', 'procesado', 'comentarios_clasificado_parte01_clasico.csv')
    ]
    
    dfs = []
    for ruta in rutas:
        if os.path.exists(ruta):
            print(f"   Cargando {os.path.basename(ruta)}...")
            df = pd.read_csv(ruta, sep=';', lineterminator='\n', on_bad_lines='skip')
            
            # Limpiar nombres de columnas (eliminar espacios)
            df.columns = df.columns.str.strip().str.lower()
            
            # La columna de emoción se pudo haber importado con retorno de carro
            if 'emocion\r' in df.columns:
                df.rename(columns={'emocion\r': 'emocion'}, inplace=True)
            elif 'emocion\n' in df.columns:
                df.rename(columns={'emocion\n': 'emocion'}, inplace=True)
            
            # A veces Pandas lee columnas con espacios extra o comillas
            df.columns = [col.replace('\r', '').replace('\n', '').replace('"', '') for col in df.columns]
                
            if 'emocion' in df.columns and 'content_clasico' in df.columns:
                # Limpiar texto de la emoción
                df['emocion'] = df['emocion'].astype(str).str.strip().str.capitalize()
                
                # Reemplazar valores extraños ocasionados por comillas/saltos de linea
                df['emocion'] = df['emocion'].str.replace('"', '')
                
                # Corregir errores de tipografía
                df['emocion'] = df['emocion'].replace({'Alegria': 'Alegría'})
                
                # Filtrar nulos y vacíos
                df = df.dropna(subset=['content_clasico', 'emocion'])
                df['content_clasico'] = df['content_clasico'].astype(str)
                
                # Filtrar por clases maestras válidas (elimina ruidos como 'Añ' y vacíos)
                emociones_validas = ['Tristeza', 'Alegría', 'Miedo', 'Sorpresa']
                df = df[df['emocion'].isin(emociones_validas)]
                
                dfs.append(df)
            else:
                print(f"   [!] Faltan columnas en {os.path.basename(ruta)}. Columnas encontradas: {df.columns.tolist()}")
        else:
            print(f"   [!] Archivo no encontrado: {ruta}")
            
    if not dfs:
        raise ValueError("No se pudo cargar ningún dataset. Revisa las rutas.")
        
    df_final = pd.concat(dfs, ignore_index=True)
    return df_final

def evaluar_modelo(modelo, nombre_modelo, X_train, y_train, X_test, y_test, archivo_reporte):
    # Entrenamiento
    modelo.fit(X_train, y_train)
    
    # Predicción en el conjunto de entrenamiento (resampled)
    y_pred_train = modelo.predict(X_train)
    
    # Predicción en el conjunto de prueba
    y_pred_test = modelo.predict(X_test)
    
    # Generar Matriz de Confusión para Prueba
    cm = confusion_matrix(y_test, y_pred_test, labels=modelo.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=modelo.classes_, yticklabels=modelo.classes_)
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    
    nombre_archivo_cm = f'cm_{nombre_modelo.lower().replace(" ", "_").replace("á", "a")}.png'
    ruta_cm = os.path.join(RESULTADOS_DIR, nombre_archivo_cm)
    plt.savefig(ruta_cm, dpi=300)
    plt.close()
    
    # Escribir resultados
    with open(archivo_reporte, 'a', encoding='utf-8') as f:
        f.write(f"## {nombre_modelo}\n\n")
        
        f.write("### Resultados en el Conjunto de Entrenamiento (SMOTE)\n")
        f.write(f"- **Accuracy:** {accuracy_score(y_train, y_pred_train):.4f}\n")
        f.write(f"- **Precision (macro):** {precision_score(y_train, y_pred_train, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **Recall (macro):** {recall_score(y_train, y_pred_train, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **F1-Score (macro):** {f1_score(y_train, y_pred_train, average='macro', zero_division=0):.4f}\n\n")
        f.write("```text\n")
        f.write(classification_report(y_train, y_pred_train, zero_division=0))
        f.write("\n```\n\n")
        
        f.write("### Resultados en el Conjunto de Prueba (30%)\n")
        f.write(f"- **Accuracy:** {accuracy_score(y_test, y_pred_test):.4f}\n")
        f.write(f"- **Precision (macro):** {precision_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **Recall (macro):** {recall_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **F1-Score (macro):** {f1_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}\n\n")
        f.write("```text\n")
        f.write(classification_report(y_test, y_pred_test, zero_division=0))
        f.write("\n```\n\n")
        # Ruta relativa a la carpeta resultados para MD
        f.write(f"**Matriz de Confusión:**\n\n![Matriz de Confusión {nombre_modelo}](./{nombre_archivo_cm})\n\n")

def ejecutar_pipeline():
    print("1. Carga de los datos...")
    df = cargar_datasets_csv()
    
    X = df['content_clasico']
    y = df['emocion']
    
    print(f"Dataset combinado: {df.shape[0]} filas.")
    print("Distribución de clases:")
    print(y.value_counts())
    
    print("\n2. División del dataset (70% entrenamiento, 30% prueba)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("\n3. Vectorización del texto con TF-IDF (mejorado con n-gramas)...")
    # Utilizamos rango 1 a 2, y podamos las muy raras o muy frecuentes
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.85)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    print(f"Dimensiones de X_train después de vectorizar: {X_train.shape}")
    
    print("\n4. Aplicando SMOTE al conjunto de entrenamiento...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    print(f"Dimensiones de X_train_res después de SMOTE: {X_train_res.shape}")
    print("Nueva distribución de clases en entrenamiento:")
    print(pd.Series(y_train_res).value_counts())
    
    modelos = {
        'SVM': SVC(kernel='linear', random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=5, random_state=42)
    }
    
    archivo_reporte = os.path.join(RESULTADOS_DIR, 'reporte_modelos_clasicos.md')
    
    # Limpiar archivo si existe
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Evaluación de Modelos Clásicos\n\n")
        f.write("Resultados de las métricas de rendimiento tras aplicar mejoras contra el sobreajuste (Vectorización Tfidf ajustada, SMOTE para balanceo y restricciones de profundidad en Random Forest).\n\n")
    
    print("\n5. Entrenamiento y evaluación de los modelos...")
    for nombre, modelo in modelos.items():
        print(f"   - Procesando {nombre}...")
        evaluar_modelo(modelo, nombre, X_train_res, y_train_res, X_test, y_test, archivo_reporte)
        
    print(f"\nProceso completado. Reporte guardado en: {archivo_reporte}")

if __name__ == "__main__":
    ejecutar_pipeline()
