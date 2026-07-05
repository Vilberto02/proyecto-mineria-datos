# %% [markdown]
# # Transformers para la clasificación de emociones
# 
# Curso: Minería de Datos
# 
# Proyecto: Clasificación automática de emociones en publicaciones de jóvenes peruanos

# %% [markdown]
# ### 1. Preparación del entorno

# %% [markdown]
# #### Verificar la GPU
# 
# Utilizaremos Google Colab para aprovechar su GPU y que el proceso sea mucho más rápido.

# %%
# pyrefly: ignore [missing-import]
import torch
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo detectado: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[!] NO se detectó GPU. El entrenamiento será muy lento.")

# %% [markdown]
# #### Entorno de ejecución

# %%
import os
import sys

try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()

BASE_DIR = current_dir
DATOS_DIR = os.path.join(BASE_DIR, 'datos')

# Si no está directamente ahí, busquemos un nivel arriba o en modelado
if not os.path.exists(DATOS_DIR):
    if os.path.exists(os.path.join(BASE_DIR, 'modelado', 'datos')):
        DATOS_DIR = os.path.join(BASE_DIR, 'modelado', 'datos')
        BASE_DIR = os.path.join(BASE_DIR, 'modelado')
    elif os.path.exists(os.path.join(os.path.dirname(BASE_DIR), 'datos')):
        BASE_DIR = os.path.dirname(BASE_DIR)
        DATOS_DIR = os.path.join(BASE_DIR, 'datos')

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

RESULTADOS_DIR = os.path.join(BASE_DIR, 'resultados_transformers')
os.makedirs(RESULTADOS_DIR, exist_ok=True)

print(f"Directorio base detectado: {BASE_DIR}")
print(f"Directorio de datos: {DATOS_DIR}")

# %% [markdown]
# #### Instalación de dependencias

# %%
# !pip install transformers datasets pysentimiento evaluate accelerate scikit-learn pandas matplotlib seaborn openpyxl

# %% [markdown]
# #### Importación de librerías

# %%
import os
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.preprocessing import LabelBinarizer
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments,
    EarlyStoppingCallback
)
# pyrefly: ignore [missing-import]
from accelerate import Accelerator

print("Todas las librerías se importaron correctamente.")

# %% [markdown]
# ### 2. Carga y estandarización de los datasets
# 
# Se cargan los 3 datasets preprocesados (YouTube parte 1, YouTube parte 2, TikTok) y se unifican en un solo dataframe.

# %%
import glob
import pandas as pd

archivos = [
    os.path.join(DATOS_DIR, 'dataset_limpio_final.csv')
]

dfs = []
for f in archivos:
    if os.path.exists(f):
        print(f"Cargando {os.path.basename(f)}...")
        temp_df = pd.read_csv(f, sep=';', encoding='utf-8')
        temp_df.columns = temp_df.columns.str.strip().str.replace('"', '').str.lower()
        if 'final_emocion' in temp_df.columns:
            temp_df['emocion'] = temp_df['final_emocion']
        dfs.append(temp_df)

if not dfs:
    raise ValueError("No se encontraron los datasets en modelado/datos/")

df = pd.concat(dfs, ignore_index=True)
df['emocion'] = df['emocion'].astype(str).str.strip().str.replace('"', '').str.capitalize()
df['emocion'] = df['emocion'].replace({'Alegría': 'Alegria'})

# Filtrar 6 emociones maestras
emociones_validas = ['Sorpresa', 'Miedo', 'Alegria', 'Tristeza']
df = df[df['emocion'].isin(emociones_validas)]
# Usamos la nueva columna del dataset consolidado
df = df.dropna(subset=['content_transformers', 'emocion'])

print(f"\nDataset unificado: {len(df)} registros")
print(f"Distribución de emociones:\n{df['emocion'].value_counts()}")

# %% [markdown]
# ### 3. División y preparación del conjunto de datos

# %%
# Se utiliza la columna content_transformers
textos = df['content_transformers'].astype(str).tolist()
etiquetas = df['emocion'].tolist()
clases = sorted(df['emocion'].unique().tolist())
print(f"Clases: {clases}")
total = len(etiquetas)
print(f"Total de registros: {total}")
# División: 80% entrenamiento / 20% prueba
# 1er corte: 20% prueba final
train_texts_temp, test_texts, train_labels_temp, test_labels = train_test_split(
    textos, etiquetas, test_size=0.20, random_state=42, stratify=etiquetas
)
# 2do corte: 30% del 80% → validación (~24% del total)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_texts_temp, train_labels_temp, test_size=0.30, random_state=42,
    stratify=train_labels_temp
)
print(f"\nEntrenamiento Efectivo: {len(train_texts)} registros (~{len(train_texts)/total*100:.1f}%)")
print(f"Validación:             {len(val_texts)} registros (~{len(val_texts)/total*100:.1f}%)")
print(f"Prueba y Evaluación:    {len(test_texts)} registros (~{len(test_texts)/total*100:.1f}%)")
# Mapa de etiquetas a índices
label2id = {e: i for i, e in enumerate(clases)}
id2label  = {i: e for i, e in enumerate(clases)}
print(f"\nMapa de etiquetas: {label2id}")
train_labels_ids = [label2id[l] for l in train_labels]
val_labels_ids   = [label2id[l] for l in val_labels]
test_labels_ids  = [label2id[l] for l in test_labels]

# %% [markdown]
# #### Pesos por clase (para manejar desbalance)

# %%
from sklearn.utils.class_weight import compute_class_weight

classes_array = np.array(clases)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes_array,
    y=np.array(train_labels)
)
class_weights_dict = {clases[i]: class_weights[i] for i in range(len(clases))}
pesos_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

print("Pesos por clase (balanced):")
for c, w in class_weights_dict.items():
    print(f"   {c}: {w:.4f}")

# %% [markdown]
# #### Clase Dataset de PyTorch

# %%
class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# %% [markdown]
# ## 4. Funciones de evaluación
# 
# Se definen las funciones para calcular métricas y generar los gráficos (matriz de confusión y curva ROC multiclase).

# %%
import gc

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision_macro': precision_score(labels, predictions, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, predictions, average='macro', zero_division=0),
        'f1_macro': f1_score(labels, predictions, average='macro', zero_division=0)
    }

def generar_matrices_confusion_conjuntas(y_train, y_pred_train, y_val, y_pred_val, y_test, y_pred_test, clases, nombre_modelo, filename, ruta):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Matrices de Confusión - {nombre_modelo}', fontsize=16)

    conjuntos = [
        ('Entrenamiento Efectivo', y_train, y_pred_train, axes[0]),
        ('Validación', y_val, y_pred_val, axes[1]),
        ('Prueba', y_test, y_pred_test, axes[2])
    ]

    for nombre_conjunto, y_real, y_pred, ax in conjuntos:
        cm = confusion_matrix(y_real, y_pred, labels=range(len(clases)))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=clases, yticklabels=clases, ax=ax)
        ax.set_title(nombre_conjunto)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')

    plt.tight_layout()
    ruta_completa = os.path.join(ruta, filename)
    plt.savefig(ruta_completa, dpi=300)
    plt.close()
    return ruta_completa

def generar_roc_curvas_conjuntas(y_train, y_scores_train, y_val, y_scores_val, y_test, y_scores_test, clases, nombre_modelo, filename, ruta):
    lb = LabelBinarizer()
    lb.fit(y_train)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'Curvas ROC - {nombre_modelo}', fontsize=16)
    
    auc_dict_ret = {}

    conjuntos = [
        ('Entrenamiento Efectivo', y_train, y_scores_train, axes[0]),
        ('Validación', y_val, y_scores_val, axes[1]),
        ('Prueba', y_test, y_scores_test, axes[2])
    ]

    for nombre_conjunto, y_real, y_scores, ax in conjuntos:
        y_true_bin = lb.transform(y_real)
        
        fpr_dict, tpr_dict, roc_auc = {}, {}, {}
        for i, clase in enumerate(clases):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr_dict[i], tpr_dict[i])
            ax.plot(fpr_dict[i], tpr_dict[i], lw=2,
                     label=f'{clase} (AUC = {roc_auc[i]:.3f})')
                     
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(clases))]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(len(clases)):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= len(clases)
        macro_auc = auc(all_fpr, mean_tpr)
        
        ax.plot(all_fpr, mean_tpr, 'k--', lw=3,
                 label=f'Macro-promedio (AUC = {macro_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'gray', linestyle=':', lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title(nombre_conjunto)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
        
        auc_dict_ret[nombre_conjunto] = {'macro': macro_auc, 'clases': roc_auc}

    plt.tight_layout()
    ruta_completa = os.path.join(ruta, filename)
    plt.savefig(ruta_completa, dpi=300)
    plt.close()
    return ruta_completa, auc_dict_ret

def reporte_por_clase(y_true, y_pred, clases):
    report = classification_report(y_true, y_pred, labels=range(len(clases)),
                                   target_names=clases, zero_division=0, output_dict=True)
    return report

def guardar_reporte_markdown(filename, ruta, nombre_modelo, metricas_totales, cm_path, roc_path,
                             auc_dict_ret, reportes_totales, clases):
    ruta_completa = os.path.join(ruta, filename)
    cm_rel = cm_path.replace('\\\\', '/')
    roc_rel = roc_path.replace('\\\\', '/')

    with open(ruta_completa, 'a', encoding='utf-8') as f:
        f.write(f"## {nombre_modelo}\n\n")
        
        for nombre_conjunto in ['Entrenamiento Efectivo', 'Validación', 'Prueba']:
            if nombre_conjunto not in metricas_totales: continue
            
            f.write(f"### Resultados en {nombre_conjunto}\n")
            f.write(f"- **Accuracy:** {metricas_totales[nombre_conjunto]['accuracy']:.4f}\n")
            f.write(f"- **Precision (macro):** {metricas_totales[nombre_conjunto]['precision_macro']:.4f}\\n")
            f.write(f"- **Recall (macro):** {metricas_totales[nombre_conjunto]['recall_macro']:.4f}\n")
            f.write(f"- **F1-Score (macro):** {metricas_totales[nombre_conjunto]['f1_macro']:.4f}\n\n")

            f.write(f"**AUC ROC ({nombre_conjunto}):**\n")
            f.write(f"- Macro-promedio: {auc_dict_ret[nombre_conjunto]['macro']:.4f}\n")
            for i, clase in enumerate(clases):
                f.write(f"- {clase}: {auc_dict_ret[nombre_conjunto]['clases'][i]:.4f}\n")
            f.write("\n")

            f.write(f"**Reporte por Clase ({nombre_conjunto})**\n\n")
            f.write("| Clase | Precisión | Recall | F1-Score | Soporte |\n")
            f.write("|-------|-----------|--------|----------|---------|\n")
            for clase in clases:
                r = reportes_totales[nombre_conjunto][clase]
                f.write(f"| {clase} | {r['precision']:.4f} | {r['recall']:.4f} | {r['f1-score']:.4f} | {int(r['support'])} |\n")
            f.write("\n")

        f.write("### Gráficas Conjuntas\n\n")
        f.write(f"**Matrices de Confusión:**\n\n![Matrices de Confusión {nombre_modelo}](./{os.path.basename(cm_path)})\n\n")
        f.write(f"**Curvas ROC:**\n\n![Curvas ROC {nombre_modelo}](./{os.path.basename(roc_path)})\n\n")
        f.write("---\n\n")

    print(f"Reporte actualizado: {ruta_completa}")
    return ruta_completa


# %% [markdown]
# #### Hiperparametros compartidos

# %%
MAX_LENGTH   = 128
BATCH_SIZE   = 16
EPOCHS       = 5       # early stopping controla el límite real
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 200

# %% [markdown]
# ### 5. Entrenamiento: BETO (BERT para corpus en español)
# 
# Modelo: [`dccuchile/bert-base-spanish-wwm-cased`](https://huggingface.co/dccuchile/bert-base-spanish-wwm-cased)

# %%
MODELO_BETO = 'dccuchile/bert-base-spanish-wwm-cased'
LEARNING_RATE_BETO = 3e-5

print(f"Configuración del Modelo Base:")
print(f"   Modelo: {MODELO_BETO}")
print(f"   Max length: {MAX_LENGTH}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: {LEARNING_RATE_BETO}")
print(f"   Weight decay: {WEIGHT_DECAY}")
print(f"   Warmup steps: {WARMUP_STEPS}")

# %%
print("Cargando tokenizer...")
tokenizer_beto = AutoTokenizer.from_pretrained(MODELO_BETO)

print("Tokenizando textos de entrenamiento efectivo...")
train_encodings_beto = tokenizer_beto(
    train_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors=None
)

print("Tokenizando textos de prueba...")
test_encodings_beto = tokenizer_beto(
    test_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors=None
)

print("Tokenizando textos de prueba y validacion...")
val_encodings_beto = tokenizer_beto(
    val_texts,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors=None
)

train_dataset_beto = EmotionDataset(train_encodings_beto, train_labels_ids)
test_dataset_beto = EmotionDataset(test_encodings_beto, test_labels_ids)
val_dataset_beto = EmotionDataset(val_encodings_beto, val_labels_ids)

print(f"Train dataset: {len(train_dataset_beto)} muestras")
print(f"Test dataset: {len(test_dataset_beto)} muestras")
print(f"Val dataset: {len(val_dataset_beto)} muestras")

# %% [markdown]
# #### Función para entrenar los modelos

# %%

def freeze_bottom_layers(model, num_layers_to_freeze=6):
    if hasattr(model, 'bert'):
        for param in model.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(min(num_layers_to_freeze, len(model.bert.encoder.layer))):
            for param in model.bert.encoder.layer[i].parameters():
                param.requires_grad = False
    elif hasattr(model, 'roberta'):
        for param in model.roberta.embeddings.parameters():
            param.requires_grad = False
        for i in range(min(num_layers_to_freeze, len(model.roberta.encoder.layer))):
            for param in model.roberta.encoder.layer[i].parameters():
                param.requires_grad = False

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=pesos_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

print("CustomTrainer implentada.")

# %% [markdown]
# #### Entrenamiento del modelo

# %%
ruta_reporte = os.path.join(RESULTADOS_DIR, 'reporte_transformers.md')
# Limpiamos el reporte al inicio de las evaluaciones
with open(ruta_reporte, 'w', encoding='utf-8') as f:
    f.write('# Reporte de evaluación de modelos Transformer\n\n')

print("\n--- Entrenando modelo BETO Final ---")
print("Cargando modelo pre-entrenado BETO...")
model_beto = AutoModelForSequenceClassification.from_pretrained(
    MODELO_BETO,
    num_labels=len(clases),
    id2label=id2label,
    label2id=label2id
).to(device)
freeze_bottom_layers(model_beto)

training_args_beto = TrainingArguments(
    output_dir='./resultados_beto',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir='./logs_beto',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    report_to="none"
)

trainer_beto = CustomTrainer(
    model=model_beto,
    args=training_args_beto,
    train_dataset=train_dataset_beto,
    eval_dataset=val_dataset_beto,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer_beto.train()

# %% [markdown]
# #### Evaluación del modelo

# %%
print("Evaluando BETO en los 3 conjuntos...")

# Entrenamiento Efectivo
beto_train_preds  = trainer_beto.predict(train_dataset_beto)
beto_train_logits = beto_train_preds.predictions
beto_train_ids    = np.argmax(beto_train_logits, axis=-1)
beto_train_metrics = {
    'accuracy':        accuracy_score(train_labels_ids, beto_train_ids),
    'precision_macro': precision_score(train_labels_ids, beto_train_ids, average='macro', zero_division=0),
    'recall_macro':    recall_score(train_labels_ids, beto_train_ids, average='macro', zero_division=0),
    'f1_macro':        f1_score(train_labels_ids, beto_train_ids, average='macro', zero_division=0),
}
print(f"[Train] Accuracy: {beto_train_metrics['accuracy']:.4f} | F1: {beto_train_metrics['f1_macro']:.4f}")

# Validación
beto_val_preds  = trainer_beto.predict(val_dataset_beto)
beto_val_logits = beto_val_preds.predictions
beto_val_ids    = np.argmax(beto_val_logits, axis=-1)
beto_val_metrics = {
    'accuracy':        accuracy_score(val_labels_ids, beto_val_ids),
    'precision_macro': precision_score(val_labels_ids, beto_val_ids, average='macro', zero_division=0),
    'recall_macro':    recall_score(val_labels_ids, beto_val_ids, average='macro', zero_division=0),
    'f1_macro':        f1_score(val_labels_ids, beto_val_ids, average='macro', zero_division=0),
}
print(f"[Val]   Accuracy: {beto_val_metrics['accuracy']:.4f} | F1: {beto_val_metrics['f1_macro']:.4f}")

# Prueba y Evaluación
beto_test_preds = trainer_beto.predict(test_dataset_beto)
beto_logits     = beto_test_preds.predictions
beto_preds_ids  = np.argmax(beto_logits, axis=-1)
beto_metrics = {
    'accuracy':        accuracy_score(test_labels_ids, beto_preds_ids),
    'precision_macro': precision_score(test_labels_ids, beto_preds_ids, average='macro', zero_division=0),
    'recall_macro':    recall_score(test_labels_ids, beto_preds_ids, average='macro', zero_division=0),
    'f1_macro':        f1_score(test_labels_ids, beto_preds_ids, average='macro', zero_division=0),
}
print(f"[Test]  Accuracy: {beto_metrics['accuracy']:.4f} | F1: {beto_metrics['f1_macro']:.4f}")

# Probabilidades para curvas ROC
beto_train_probas = torch.nn.functional.softmax(torch.tensor(beto_train_logits), dim=-1).numpy()
beto_val_probas   = torch.nn.functional.softmax(torch.tensor(beto_val_logits),   dim=-1).numpy()
beto_probas       = torch.nn.functional.softmax(torch.tensor(beto_logits),        dim=-1).numpy()

# Matrices de Confusión Conjuntas
beto_cm_path = generar_matrices_confusion_conjuntas(
    train_labels_ids, beto_train_ids,
    val_labels_ids, beto_val_ids,
    test_labels_ids, beto_preds_ids,
    clases, 'BETO', 'cm_conjunta_beto.png', RESULTADOS_DIR
)

# Curvas ROC Conjuntas
beto_roc_path, beto_auc_dict_ret = generar_roc_curvas_conjuntas(
    train_labels_ids, beto_train_probas,
    val_labels_ids, beto_val_probas,
    test_labels_ids, beto_probas,
    clases, 'BETO', 'roc_conjunta_beto.png', RESULTADOS_DIR
)

# Reportes por clase
beto_report_train = reporte_por_clase(train_labels_ids, beto_train_ids, clases)
beto_report_val   = reporte_por_clase(val_labels_ids,   beto_val_ids,   clases)
beto_report       = reporte_por_clase(test_labels_ids,  beto_preds_ids, clases)

beto_metrics_totales = {
    'Entrenamiento Efectivo': beto_train_metrics,
    'Validación': beto_val_metrics,
    'Prueba': beto_metrics
}

beto_reportes_totales = {
    'Entrenamiento Efectivo': beto_report_train,
    'Validación': beto_report_val,
    'Prueba': beto_report
}

guardar_reporte_markdown(
    'reporte_transformers.md', RESULTADOS_DIR, 'BETO', beto_metrics_totales,
    beto_cm_path, beto_roc_path, beto_auc_dict_ret, beto_reportes_totales, clases
)

# %% [markdown]
# ### 6. Entrenamiento: RoBERTuito (RoBERTa en español)
# 
# Modelo: [`pysentimiento/robertuito-base-uncased`](https://huggingface.co/pysentimiento/robertuito-base-uncased)

# %%
# pyrefly: ignore [missing-import]
from pysentimiento.preprocessing import preprocess_tweet

MODELO_ROBERTA = 'pysentimiento/robertuito-base-uncased'
LEARNING_RATE_ROBERTA = 1e-5

print(f"Configuración del Modelo Base:")
print(f"   Modelo: {MODELO_ROBERTA}")
print(f"   Max length: {MAX_LENGTH}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Epochs: {EPOCHS}")
print(f"   Learning rate: {LEARNING_RATE_ROBERTA}")
print(f"   Weight decay: {WEIGHT_DECAY}")
print(f"   Warmup steps: {WARMUP_STEPS}")

# %%
print("Cargando tokenizer RoBERTa...")
tokenizer_roberta = AutoTokenizer.from_pretrained('pysentimiento/robertuito-base-uncased')

train_texts_proc = [preprocess_tweet(t) for t in train_texts]
test_texts_proc  = [preprocess_tweet(t) for t in test_texts]

print("Tokenizando textos de entrenamiento efectivo...")
train_encodings_roberta = tokenizer_roberta(
    train_texts_proc,   # textos preprocesados
    truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors=None
)

print("Tokenizando textos de validacion...")
test_encodings_roberta = tokenizer_roberta(
    test_texts_proc,    # textos preprocesados
    truncation=True, padding=True, max_length=MAX_LENGTH, return_tensors=None
)

print("Tokenizando textos de prueba y validacion...")
val_texts_proc = [preprocess_tweet(t) for t in val_texts]
val_encodings_roberta = tokenizer_roberta(
    val_texts_proc,
    truncation=True,
    padding=True,
    max_length=MAX_LENGTH,
    return_tensors=None
)

train_dataset_roberta = EmotionDataset(train_encodings_roberta, train_labels_ids)
test_dataset_roberta = EmotionDataset(test_encodings_roberta, test_labels_ids)
val_dataset_roberta = EmotionDataset(val_encodings_roberta, val_labels_ids)

print(f"Train dataset: {len(train_dataset_roberta)} muestras")
print(f"Test dataset: {len(test_dataset_roberta)} muestras")
print(f"Val dataset: {len(val_dataset_roberta)} muestras")

# %%
ruta_reporte = os.path.join(RESULTADOS_DIR, 'reporte_transformers.md')

print("\nEntrenando modelo RoBERTuito Final")
print("Cargando modelo pre-entrenado RoBERTuito...")
model_roberta = AutoModelForSequenceClassification.from_pretrained(
    MODELO_ROBERTA,
    num_labels=len(clases),
    id2label=id2label,
    label2id=label2id
).to(device)
freeze_bottom_layers(model_roberta)

training_args_roberta = TrainingArguments(
    output_dir='./resultados_roberta',
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    weight_decay=WEIGHT_DECAY,
    logging_dir='./logs_roberta',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    report_to="none"
)

trainer_roberta = CustomTrainer(
    model=model_roberta,
    args=training_args_roberta,
    train_dataset=train_dataset_roberta,
    eval_dataset=val_dataset_roberta,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer_roberta.train()

# %%
print("Evaluando RoBERTa en los 3 conjuntos...")

# Entrenamiento Efectivo
roberta_train_preds  = trainer_roberta.predict(train_dataset_roberta)
roberta_train_logits = roberta_train_preds.predictions
roberta_train_ids    = np.argmax(roberta_train_logits, axis=-1)
roberta_train_metrics = {
    'accuracy':        accuracy_score(train_labels_ids, roberta_train_ids),
    'precision_macro': precision_score(train_labels_ids, roberta_train_ids, average='macro', zero_division=0),
    'recall_macro':    recall_score(train_labels_ids, roberta_train_ids, average='macro', zero_division=0),
    'f1_macro':        f1_score(train_labels_ids, roberta_train_ids, average='macro', zero_division=0),
}
print(f"[Train] Accuracy: {roberta_train_metrics['accuracy']:.4f} | F1: {roberta_train_metrics['f1_macro']:.4f}")

# Validación
roberta_val_preds  = trainer_roberta.predict(val_dataset_roberta)
roberta_val_logits = roberta_val_preds.predictions
roberta_val_ids    = np.argmax(roberta_val_logits, axis=-1)
roberta_val_metrics = {
    'accuracy':        accuracy_score(val_labels_ids, roberta_val_ids),
    'precision_macro': precision_score(val_labels_ids, roberta_val_ids, average='macro', zero_division=0),
    'recall_macro':    recall_score(val_labels_ids, roberta_val_ids, average='macro', zero_division=0),
    'f1_macro':        f1_score(val_labels_ids, roberta_val_ids, average='macro', zero_division=0),
}
print(f"[Val]   Accuracy: {roberta_val_metrics['accuracy']:.4f} | F1: {roberta_val_metrics['f1_macro']:.4f}")

# Prueba y Evaluación
roberta_test_preds = trainer_roberta.predict(test_dataset_roberta)
roberta_logits     = roberta_test_preds.predictions
roberta_preds_ids  = np.argmax(roberta_logits, axis=-1)
roberta_metrics = {
    'accuracy':        accuracy_score(test_labels_ids, roberta_preds_ids),
    'precision_macro': precision_score(test_labels_ids, roberta_preds_ids, average='macro', zero_division=0),
    'recall_macro':    recall_score(test_labels_ids, roberta_preds_ids, average='macro', zero_division=0),
    'f1_macro':        f1_score(test_labels_ids, roberta_preds_ids, average='macro', zero_division=0),
}
print(f"[Test]  Accuracy: {roberta_metrics['accuracy']:.4f} | F1: {roberta_metrics['f1_macro']:.4f}")

# Probabilidades para curvas ROC
roberta_train_probas = torch.nn.functional.softmax(torch.tensor(roberta_train_logits), dim=-1).numpy()
roberta_val_probas   = torch.nn.functional.softmax(torch.tensor(roberta_val_logits),   dim=-1).numpy()
roberta_probas       = torch.nn.functional.softmax(torch.tensor(roberta_logits),        dim=-1).numpy()

# Matrices de Confusión Conjuntas
roberta_cm_path = generar_matrices_confusion_conjuntas(
    train_labels_ids, roberta_train_ids,
    val_labels_ids, roberta_val_ids,
    test_labels_ids, roberta_preds_ids,
    clases, 'RoBERTuito', 'cm_conjunta_roberta.png', RESULTADOS_DIR
)

# Curvas ROC Conjuntas
roberta_roc_path, roberta_auc_dict_ret = generar_roc_curvas_conjuntas(
    train_labels_ids, roberta_train_probas,
    val_labels_ids, roberta_val_probas,
    test_labels_ids, roberta_probas,
    clases, 'RoBERTuito', 'roc_conjunta_roberta.png', RESULTADOS_DIR
)

# Reportes por clase
roberta_report_train = reporte_por_clase(train_labels_ids, roberta_train_ids, clases)
roberta_report_val   = reporte_por_clase(val_labels_ids,   roberta_val_ids,   clases)
roberta_report       = reporte_por_clase(test_labels_ids,  roberta_preds_ids, clases)

roberta_metrics_totales = {
    'Entrenamiento Efectivo': roberta_train_metrics,
    'Validación': roberta_val_metrics,
    'Prueba': roberta_metrics
}

roberta_reportes_totales = {
    'Entrenamiento Efectivo': roberta_report_train,
    'Validación': roberta_report_val,
    'Prueba': roberta_report
}

guardar_reporte_markdown(
    'reporte_transformers.md', RESULTADOS_DIR, 'RoBERTuito', roberta_metrics_totales,
    roberta_cm_path, roberta_roc_path, roberta_auc_dict_ret, roberta_reportes_totales, clases
)

# %% [markdown]
# 
# ### 7. Exportación del reporte

# %%
# La limpieza del reporte se hace antes de evaluar BETO (CV).
