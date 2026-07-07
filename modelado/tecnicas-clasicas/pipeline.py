import os
import sys
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
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

from sklearn.pipeline import Pipeline

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

def graficar_roc_curvas_conjuntas(nombre_modelo, mejor_modelo, X_train, y_train, X_val, y_val, X_test, y_test, clases):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Curvas ROC Multiclase (OvR) — {nombre_modelo}', fontsize=16)
    
    auc_dict_ret = {}
    
    conjuntos = [
        ('Entrenamiento Efectivo', X_train, y_train, axes[0]),
        ('Validación', X_val, y_val, axes[1]),
        ('Prueba', X_test, y_test, axes[2])
    ]
    
    colores = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
    
    for nombre_conjunto, X_datos, y_datos, ax in conjuntos:
        y_bin = label_binarize(y_datos, classes=clases)
        y_score = mejor_modelo.predict_proba(X_datos)
        
        fpr, tpr, roc_auc = {}, {}, {}
        for i, clase in enumerate(clases):
            fpr[clase], tpr[clase], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[clase] = auc(fpr[clase], tpr[clase])
            ax.plot(fpr[clase], tpr[clase], color=colores[i % len(colores)], lw=2,
                     label=f'{clase} (AUC = {roc_auc[clase]:.3f})')
                     
        all_fpr = np.unique(np.concatenate([fpr[c] for c in clases]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in clases:
            mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
        mean_tpr /= len(clases)
        auc_macro = auc(all_fpr, mean_tpr)
        
        ax.plot(all_fpr, mean_tpr, color='black', lw=2.5, linestyle='--',
                 label=f'Macro-avg (AUC = {auc_macro:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(nombre_conjunto)
        ax.legend(loc='lower right', fontsize=8)
        
        auc_dict_ret[nombre_conjunto] = {'macro': auc_macro, 'clases': roc_auc}

    plt.tight_layout()
    nombre_base = nombre_modelo.lower().replace(' ', '_').replace('á', 'a')
    nombre_roc = f'roc_conjunta_{nombre_base}.png'
    ruta_roc = os.path.join(RESULTADOS_DIR, nombre_roc)
    plt.savefig(ruta_roc, dpi=300)
    plt.close()

    return auc_dict_ret, nombre_roc


def graficar_matrices_confusion_conjuntas(nombre_modelo, y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, clases):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Matrices de Confusión - {nombre_modelo}', fontsize=16)

    conjuntos = [
        ('Entrenamiento Efectivo', y_train, y_pred_train, axes[0]),
        ('Validación', y_val, y_pred_val, axes[1]),
        ('Prueba', y_test, y_pred_test, axes[2])
    ]

    for nombre_conjunto, y_real, y_pred, ax in conjuntos:
        cm = confusion_matrix(y_real, y_pred, labels=clases)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=clases, yticklabels=clases, ax=ax)
        ax.set_title(nombre_conjunto)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')

    plt.tight_layout()
    nombre_base = nombre_modelo.lower().replace(' ', '_').replace('á', 'a')
    nombre_archivo_cm = f'cm_conjunta_{nombre_base}.png'
    ruta_cm = os.path.join(RESULTADOS_DIR, nombre_archivo_cm)
    plt.savefig(ruta_cm, dpi=300)
    plt.close()

    return nombre_archivo_cm

def evaluar_modelo(nombre_modelo, pipeline,
                     X_train, y_train, X_val, y_val, X_test, y_test,
                     archivo_reporte):
    print(f"   -> Entrenando {nombre_modelo}...")
    
    pipeline.fit(X_train, y_train)
    mejor_modelo = pipeline

    clases = list(mejor_modelo.classes_)

    y_pred_train = mejor_modelo.predict(X_train)
    y_pred_val   = mejor_modelo.predict(X_val)
    y_pred_test  = mejor_modelo.predict(X_test)

    nombre_cm_conjunta = graficar_matrices_confusion_conjuntas(
        nombre_modelo, y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, clases
    )

    auc_dict_ret, nombre_roc_conjunta = graficar_roc_curvas_conjuntas(
        nombre_modelo, mejor_modelo, X_train, y_train, X_val, y_val, X_test, y_test, clases
    )

    with open(archivo_reporte, 'a', encoding='utf-8') as f:
        f.write(f"## {nombre_modelo}\n\n")

        # — Entrenamiento Efectivo (~56%) —
        f.write("### Resultados en el Conjunto de Entrenamiento Efectivo (~56%)\n")
        f.write(f"- **Accuracy:** {accuracy_score(y_train, y_pred_train):.4f}\n")
        f.write(f"- **Precision (macro):** {precision_score(y_train, y_pred_train, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **Recall (macro):** {recall_score(y_train, y_pred_train, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **F1-Score (macro):** {f1_score(y_train, y_pred_train, average='macro', zero_division=0):.4f}\n\n")
        f.write("```text\n")
        f.write(classification_report(y_train, y_pred_train, zero_division=0))
        f.write("\n```\n\n")
        
        f.write("**AUC ROC (Entrenamiento):**\n")
        f.write(f"- Macro-average: {auc_dict_ret['Entrenamiento Efectivo']['macro']:.4f}\n")
        for clase, auc_val in auc_dict_ret['Entrenamiento Efectivo']['clases'].items():
            f.write(f"- {clase}: {auc_val:.4f}\n")
        f.write("\n")

        # — Validación (~24%) —
        f.write("### Resultados en el Conjunto de Validación (~24%)\n")
        f.write(f"- **Accuracy:** {accuracy_score(y_val, y_pred_val):.4f}\n")
        f.write(f"- **Precision (macro):** {precision_score(y_val, y_pred_val, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **Recall (macro):** {recall_score(y_val, y_pred_val, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **F1-Score (macro):** {f1_score(y_val, y_pred_val, average='macro', zero_division=0):.4f}\n\n")
        f.write("```text\n")
        f.write(classification_report(y_val, y_pred_val, zero_division=0))
        f.write("\n```\n\n")

        f.write("**AUC ROC (Validación):**\n")
        f.write(f"- Macro-average: {auc_dict_ret['Validación']['macro']:.4f}\n")
        for clase, auc_val in auc_dict_ret['Validación']['clases'].items():
            f.write(f"- {clase}: {auc_val:.4f}\n")
        f.write("\n")

        # — Prueba y Evaluación (20%) —
        f.write("### Resultados en el Conjunto de Prueba y Evaluación (20%)\n")
        f.write(f"- **Accuracy:** {accuracy_score(y_test, y_pred_test):.4f}\n")
        f.write(f"- **Precision (macro):** {precision_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **Recall (macro):** {recall_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **F1-Score (macro):** {f1_score(y_test, y_pred_test, average='macro', zero_division=0):.4f}\n\n")
        f.write("```text\n")
        f.write(classification_report(y_test, y_pred_test, zero_division=0))
        f.write("\n```\n\n")

        f.write("**AUC ROC (Prueba):**\n")
        f.write(f"- Macro-average: {auc_dict_ret['Prueba']['macro']:.4f}\n")
        for clase, auc_val in auc_dict_ret['Prueba']['clases'].items():
            f.write(f"- {clase}: {auc_val:.4f}\n")
        f.write("\n")
        
        # — Gráficas Conjuntas —
        f.write("### Gráficas de Evaluación Conjuntas\n\n")
        f.write(f"**Matrices de Confusión:**\n\n![Matrices de Confusión {nombre_modelo}](./{nombre_cm_conjunta})\n\n")
        f.write(f"**Curvas ROC:**\n\n![Curvas ROC {nombre_modelo}](./{nombre_roc_conjunta})\n\n")
        f.write("---\n\n")

def ejecutar_pipeline():
    print("1. Carga de los datos...")
    df = cargar_datasets()
    
    print("2. Preprocesamiento de los datos de la columna emocion...")
    X, y = preprocesar_datos(df)

    total = len(y)
    print(f"\n   Total de registros (4 emociones): {total}")
    
    # División de datos: 80% entrenamiento / 20% prueba
    # Luego, del 80%: 70% entrenamiento efectivo / 30% validación
    print("3. División del dataset:")
    print("   - 80% : Entrenamiento (70% efectivo + 30% validación)")
    print("   - 20% : Prueba y Evaluación")

    # 1er corte: separar el 20% de prueba
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # 2do corte: del 80% restante, 30% para validación (= 24% del total)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.30, random_state=42, stratify=y_temp
    )

    print(f"   Entrenamiento Efectivo: {len(X_train)} registros (~{len(X_train)/total*100:.1f}%)")
    print(f"   Validación:             {len(X_val)} registros (~{len(X_val)/total*100:.1f}%)")
    print(f"   Prueba y Evaluación:    {len(X_test)} registros (~{len(X_test)/total*100:.1f}%)")

    # Distribución por clase en cada conjunto
    emociones = sorted(y.unique())
    print("\n   Distribución por emoción en cada conjunto:")
    print(f"   {'Emoción':<14} {'Entren. Efect.':>16} {'Validación':>12} {'Prueba':>8}")
    print(f"   {'-'*54}")
    for em in emociones:
        n_train = (y_train == em).sum()
        n_val   = (y_val   == em).sum()
        n_test  = (y_test  == em).sum()
        print(f"   {em:<14} {n_train:>16} {n_val:>12} {n_test:>8}")
    print(f"   {'TOTAL':<14} {len(y_train):>16} {len(y_val):>12} {len(y_test):>8}\n")

    print("4. Configuración de Pipelines y Grillas...")
    
    # Diccionarios de configuración para cada modelo
    configuraciones = [
        {
            'nombre': 'SVM',
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=2000, min_df=5, max_df=0.85)),
                ('clf', CalibratedClassifierCV(
                    SVC(kernel='linear', C=1, class_weight='balanced', random_state=42),
                    cv=5, method='sigmoid'
                ))
            ])
        },
        {
            'nombre': 'Random Forest',
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=2000, min_df=5, max_df=0.85)),
                ('clf', RandomForestClassifier(max_depth=20, min_samples_split=5, class_weight='balanced', random_state=42))
            ])
        },
        {
            'nombre': 'Naive Bayes',
            'pipeline': Pipeline([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=2000, min_df=5, max_df=0.85)),
                ('clf', MultinomialNB())
            ])
        }
    ]
    
    archivo_reporte = os.path.join(RESULTADOS_DIR, 'reporte_modelos_clasicos.md')
    
    # Limpiar archivo si existe
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Evaluación de Modelos Clásicos\n\n")
        f.write("## División de Datos\n\n")
        f.write("| Conjunto | Registros | % del total |\n")
        f.write("|---|---|---|\n")
        f.write(f"| Entrenamiento Efectivo | {len(X_train)} | ~{len(X_train)/total*100:.1f}% |\n")
        f.write(f"| Validación | {len(X_val)} | ~{len(X_val)/total*100:.1f}% |\n")
        f.write(f"| Prueba y Evaluación | {len(X_test)} | ~{len(X_test)/total*100:.1f}% |\n")
        f.write(f"| **Total** | **{total}** | **100%** |\n\n")
        f.write("## Distribución por Emoción\n\n")
        f.write("| Emoción | Entren. Efectivo | Validación | Prueba |\n")
        f.write("|---|---|---|---|\n")
        for em in emociones:
            n_train = (y_train == em).sum()
            n_val   = (y_val   == em).sum()
            n_test  = (y_test  == em).sum()
            f.write(f"| {em} | {n_train} | {n_val} | {n_test} |\n")
        f.write(f"| **Total** | **{len(y_train)}** | **{len(y_val)}** | **{len(y_test)}** |\n\n")
        f.write("---\n\n")
    
    print("5. Entrenamiento...")
    for conf in configuraciones:
        evaluar_modelo(
            conf['nombre'], 
            conf['pipeline'], 
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            archivo_reporte
        )
        
    print(f"\nProceso completado. Reporte guardado en: {archivo_reporte}")

if __name__ == "__main__":
    ejecutar_pipeline()
