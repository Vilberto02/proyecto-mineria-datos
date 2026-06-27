import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    roc_curve, auc
)

# pyrefly: ignore [missing-import]
from imblearn.pipeline import Pipeline
# pyrefly: ignore [missing-import]
from imblearn.over_sampling import SMOTE

# Configuración de carpetas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from app import cargar_datasets

DATOS_DIR = os.path.join(BASE_DIR, 'datos')
RESULTADOS_DIR = os.path.join(BASE_DIR, 'resultados_tecnicas-clasicas')

os.makedirs(RESULTADOS_DIR, exist_ok=True)

def preprocesar_datos(df):
    # Asegurarnos de que existen las columnas lemas
    if 'lemas' not in df.columns or 'emocion' not in df.columns:
        raise ValueError("Faltan columnas requeridas ('lemas' o 'emocion') en el dataset.")
    
    # Eliminar filas con valores nulos
    df = df.dropna(subset=['lemas', 'emocion'])
    
    # Convertir a string
    df['lemas'] = df['lemas'].astype(str)
    
    # Estandarizar el texto de la emoción
    df['emocion'] = df['emocion'].astype(str).str.strip().str.capitalize()
    
    # Asegurarnos de usar 'Alegria' (sin tilde) como lo solicitaste
    df['emocion'] = df['emocion'].replace({'Alegría': 'Alegria'})
    
    # Filtrar las emociones
    emociones_validas = ['Sorpresa', 'Miedo', 'Alegria', 'Tristeza']
    df = df[df['emocion'].isin(emociones_validas)]
    
    return df['lemas'], df['emocion']

def graficar_roc_curva(nombre_modelo, mejor_modelo, X_test, y_test, clases):
    """
    Genera la curva ROC multiclase usando estrategia One-vs-Rest (OvR).
    Calcula una curva por clase y la macro-average.
    Exporta un PNG en RESULTADOS_DIR y devuelve el AUC macro + AUC por clase.
    """
    y_test_bin = label_binarize(y_test, classes=clases)
    y_score    = mejor_modelo.predict_proba(X_test)

    fpr, tpr, roc_auc = {}, {}, {}
    for i, clase in enumerate(clases):
        fpr[clase], tpr[clase], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[clase] = auc(fpr[clase], tpr[clase])

    # Macro-average interpolada
    all_fpr  = np.unique(np.concatenate([fpr[c] for c in clases]))
    mean_tpr = np.zeros_like(all_fpr)
    for c in clases:
        mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
    mean_tpr /= len(clases)
    auc_macro = auc(all_fpr, mean_tpr)

    # Gráfico
    colores = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    plt.figure(figsize=(9, 7))
    for i, (clase, color) in enumerate(zip(clases, colores)):
        plt.plot(fpr[clase], tpr[clase], color=color, lw=2,
                 label=f'{clase} (AUC = {roc_auc[clase]:.3f})')
    plt.plot(all_fpr, mean_tpr, color='black', lw=2.5, linestyle='--',
             label=f'Macro-avg (AUC = {auc_macro:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5,
             label='Clasificador aleatorio')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title(f'Curva ROC Multiclase (OvR) — {nombre_modelo}', fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    nombre_roc = f'roc_{nombre_modelo.lower().replace(" ", "_").replace("á","a")}.png'
    ruta_roc   = os.path.join(RESULTADOS_DIR, nombre_roc)
    plt.savefig(ruta_roc, dpi=300)
    plt.close()

    return auc_macro, roc_auc, nombre_roc

def evaluar_modelo_grid(nombre_modelo, pipeline, param_grid, X_train, y_train, X_test, y_test, archivo_reporte, skf):
    print(f"   -> Entrenando {nombre_modelo}...")
    
    if param_grid:
        print(f"      (Buscando mejores hiperparámetros con GridSearchCV...)")
        grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring='f1_macro', n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        mejor_modelo = grid.best_estimator_
        mejores_params = grid.best_params_
        print(f"      Mejores hiperparámetros: {mejores_params}")
    else:
        pipeline.fit(X_train, y_train)
        mejor_modelo = pipeline
        mejores_params = "Por defecto"
        
    y_pred_train = mejor_modelo.predict(X_train)
    y_pred_test = mejor_modelo.predict(X_test)
    
    # Generar Matriz de Confusión
    cm = confusion_matrix(y_test, y_pred_test, labels=mejor_modelo.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=mejor_modelo.classes_, yticklabels=mejor_modelo.classes_)
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    
    nombre_archivo_cm = f'cm_{nombre_modelo.lower().replace(" ", "_").replace("á", "a")}.png'
    ruta_cm = os.path.join(RESULTADOS_DIR, nombre_archivo_cm)
    plt.savefig(ruta_cm, dpi=300)
    plt.close()
    
    # Generar Curva ROC multiclase (One-vs-Rest)
    clases = list(mejor_modelo.classes_)
    auc_macro, roc_auc_por_clase, nombre_roc = graficar_roc_curva(
        nombre_modelo, mejor_modelo, X_test, y_test, clases
    )
    
    # Escribir resultados
    with open(archivo_reporte, 'a', encoding='utf-8') as f:
        f.write(f"## {nombre_modelo}\n\n")
        
        if param_grid:
            f.write(f"**Mejores Hiperparámetros Encontrados:** `{mejores_params}`\n\n")
            
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
        f.write(f"**Matriz de Confusión:**\n\n![Matriz de Confusión {nombre_modelo}](./cm_{nombre_modelo.lower().replace(' ', '_').replace('á','a')}.png)\n\n")
        
        # Sección ROC / AUC
        f.write("### Curva ROC y AUC (One-vs-Rest)\n\n")
        f.write(f"- **AUC Macro-average:** {auc_macro:.4f}\n")
        for clase, auc_val in roc_auc_por_clase.items():
            f.write(f"- **AUC {clase}:** {auc_val:.4f}\n")
        f.write(f"\n![Curva ROC {nombre_modelo}](./{nombre_roc})\n\n")
        f.write("---\n\n")

def ejecutar_pipeline():
    print("1. Carga de los datos...")
    df = cargar_datasets()
    
    print("2. Preprocesamiento de los datos de la columna emocion...")
    X, y = preprocesar_datos(df)
    
    print("3. División del dataset (70% entrenamiento, 30% prueba)...")
    # Nota: No transformamos TF-IDF aquí. Dejamos el texto crudo para que el Pipeline
    # calcule el TF-IDF dentro del fold cruzado y no se fugue información.
    X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print("4. Configuración de Pipelines y Grillas...")
    
    # Diccionarios de configuración para cada modelo
    configuraciones = [
        {
            'nombre': 'SVM',
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.85)),
                ('smote', SMOTE(random_state=42)),
                ('clf', CalibratedClassifierCV(
                    SVC(kernel='linear', class_weight='balanced', random_state=42),
                    cv=3, method='sigmoid'
                ))
            ]),
            'param_grid': {
                'clf__estimator__C': [0.1, 1, 10]
            }
        },
        {
            'nombre': 'Random Forest',
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.85)),
                ('smote', SMOTE(random_state=42)),
                ('clf', RandomForestClassifier(class_weight='balanced', random_state=42))
            ]),
            'param_grid': {
                'clf__max_depth': [10, 20, 30],
                'clf__min_samples_split': [2, 5, 10]
            }
        },
        {
            'nombre': 'Naive Bayes',
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.85)),
                ('smote', SMOTE(random_state=42)),
                ('clf', MultinomialNB())
            ]),
            'param_grid': {} # Ejecución normal sin GridSearchCV
        }
    ]
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    archivo_reporte = os.path.join(RESULTADOS_DIR, 'reporte_modelos_clasicos.md')
    
    # Limpiar archivo si existe
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Evaluación de Modelos Clásicos\n\n")
        f.write("Evaluación robusta utilizando `imblearn.pipeline`, GridSearchCV y SMOTE para evitar Data Leakage y mitigar sobreajuste.\n\n")
    
    print("5. Entrenamiento y optimización de hiperparámetros...")
    for conf in configuraciones:
        evaluar_modelo_grid(
            conf['nombre'], 
            conf['pipeline'], 
            conf['param_grid'], 
            X_train_text, y_train, X_test_text, y_test, 
            archivo_reporte, 
            skf
        )
        
    print(f"\nProceso completado. Reporte guardado en: {archivo_reporte}")

if __name__ == "__main__":
    ejecutar_pipeline()
