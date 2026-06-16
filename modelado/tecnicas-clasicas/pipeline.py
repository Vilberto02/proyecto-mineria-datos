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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuración de carpetas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from app import cargar_datasets

DATOS_DIR = os.path.join(BASE_DIR, 'datos')
RESULTADOS_DIR = os.path.join(BASE_DIR, 'resultados')

os.makedirs(RESULTADOS_DIR, exist_ok=True)

def preprocesar_datos(df):
    # Asegurarnos de que existen las columnas lemas (realizada en el preprocesamiento del dataset)
    if 'lemas' not in df.columns or 'emocion' not in df.columns:
        raise ValueError("Faltan columnas requeridas ('lemas' o 'emocion') en el dataset.")
    
    # Eliminar filas con valores nulos en texto o emoción
    df = df.dropna(subset=['lemas', 'emocion'])
    
    # Convertir a string por si hay algún dato numérico en la columna lemas
    df['lemas'] = df['lemas'].astype(str)
    
    return df['lemas'], df['emocion']

def evaluar_modelo(modelo, nombre_modelo, X_train, y_train, X_test, y_test, archivo_reporte):
    # Entrenamiento
    modelo.fit(X_train, y_train)
    
    # Predicción en el conjunto de entrenamiento
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
        
        f.write("### Resultados en el Conjunto de Entrenamiento (70%)\n")
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
        ruta_cm_md = ruta_cm.replace('\\', '/')
        f.write(f"**Matriz de Confusión:**\n\n![Matriz de Confusión {nombre_modelo}](file:///{ruta_cm_md})\n\n")

def ejecutar_pipeline():
    print("1. Carga de los datos...")
    df = cargar_datasets()
    
    print("2. Preprocesamiento de los datos de la columna emocion...")
    X, y = preprocesar_datos(df)
    
    print("3. Divisision del dataset (70% entrenamiento, 30% prueba)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("4. Vectorizacion del texto con TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)
    
    modelos = {
        'SVM': SVC(kernel='linear', random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Árbol de Decisión': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    archivo_reporte = os.path.join(RESULTADOS_DIR, 'reporte_modelos_clasicos.md')
    
    # Limpiar archivo si existe
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Evaluación de Modelos Clásicos\n\n")
        f.write("Resultados de las métricas de rendimiento de los diferentes algoritmos clásicos (SVM, Naive Bayes, Árbol de Decisión y Random Forest).\n\n")
    
    print("5. Entrenamiento y evaluación de los modelos...")
    for nombre, modelo in modelos.items():
        print(f"   - Procesando {nombre}...")
        evaluar_modelo(modelo, nombre, X_train, y_train, X_test, y_test, archivo_reporte)
        
    print(f"\nProceso completado. Reporte guardado en: {archivo_reporte}")

if __name__ == "__main__":
    ejecutar_pipeline()
