# %% [markdown]
# # Redes neuronales para la clasificación de emociones
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

# Detección del entorno de ejecución
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

RESULTADOS_DIR = os.path.join(BASE_DIR, 'resultados_nn')
os.makedirs(RESULTADOS_DIR, exist_ok=True)

print(f"Directorio de datos: {DATOS_DIR}")
print(f"Directorio de resultados: {RESULTADOS_DIR}")


# %% [markdown]
# #### Importación de librerías

# %%
import glob
import collections
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, TensorDataset
# pyrefly: ignore [missing-import]
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Dispositivo de procesamiento detectado: {device}")

# %% [markdown]
# #### Hiperparámetros de la Red Neuronal
# 

# %%
# Valores
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 10
PATIENCE = 3 # Early stopping tolerance
LEARNING_RATE = 0.001

# %% [markdown]
# ### 2. Carga y estandarización de los datasets
# 
# Se cargan los 3 datasets preprocesados (YouTube parte 1, YouTube parte 2, TikTok) y se unifican en un solo dataframe.

# %%
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
    raise ValueError(f"No se encontraron datasets en la ruta: {DATOS_DIR}")

df = pd.concat(dfs, ignore_index=True)
df['emocion'] = df['emocion'].astype(str).str.strip().str.replace('"', '').str.capitalize()
df['emocion'] = df['emocion'].replace({'Alegría': 'Alegria'})

# Filtro de las emociones
emociones_validas = ['Sorpresa', 'Miedo', 'Alegria', 'Tristeza']
df = df[df['emocion'].isin(emociones_validas)]
df = df.dropna(subset=['lemas', 'emocion'])

print(f"\nTotal de registros consolidados: {len(df)}")
print(f"Distribución de clases:\n{df['emocion'].value_counts()}")


# %% [markdown]
# ### 3. Preparación de tensores
# 

# %%
le = LabelEncoder()
y = le.fit_transform(df['emocion'])
texts = df['lemas'].astype(str).tolist()
def build_vocab(texts, max_words=MAX_WORDS):
    words = []
    for text in texts:
        words.extend(str(text).split())
    counter = collections.Counter(words)
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(max_words - 2))}
    return vocab
def texts_to_sequences(texts, vocab):
    seqs = []
    for text in texts:
        seq = [vocab.get(w, 1) for w in str(text).split()]
        seqs.append(seq)
    return seqs
def pad_sequences(seqs, maxlen=MAX_LEN):
    padded = np.zeros((len(seqs), maxlen), dtype=int)
    for i, seq in enumerate(seqs):
        if len(seq) > maxlen:
            padded[i, :] = seq[:maxlen]
        else:
            padded[i, -len(seq):] = seq
    return padded
vocab = build_vocab(texts)
vocab_size = min(MAX_WORDS, len(vocab) + 2)
seqs = texts_to_sequences(texts, vocab)
X = pad_sequences(seqs)

# División de datos: 80% entrenamiento / 20% prueba
# Luego, del 80%: 70% entrenamiento efectivo / 30% validación
total = len(y)
print(f'Total de registros (4 emociones): {total}')
# 1er corte: separar el 20% de prueba
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
# 2do corte: del 80% restante, 30% para validación (= ~24% del total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.30, random_state=42, stratify=y_temp
)
print(f'Entrenamiento Efectivo: {len(X_train)} registros (~{len(X_train)/total*100:.1f}%)')
print(f'Validación:             {len(X_val)} registros (~{len(X_val)/total*100:.1f}%)')
print(f'Prueba y Evaluación:    {len(X_test)} registros (~{len(X_test)/total*100:.1f}%)')
X_train_t = torch.tensor(X_train, dtype=torch.long)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_val_t   = torch.tensor(X_val,   dtype=torch.long)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t,   y_val_t)
test_dataset  = TensorDataset(X_test_t,  y_test_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)
num_classes = len(le.classes_)
print(f"Clases identificadas ({num_classes}): {le.classes_}")

# %% [markdown]
# ### 4. Definición de modelos de deep learning (CNN y LSTM)
# 

# %%
# CNN 1D
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x) 
        x = x.permute(0, 2, 1) 
        x = self.conv(x)
        x = self.relu(x)
        x = torch.max(x, dim=2)[0] 
        out = self.fc(x)
        return out

# LSTM
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, (hn, cn) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

print("Modelos listos.")

# %% [markdown]
# #### Función de entrenamiento de los modelos
# 

# %%
def train_model(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"   Época {epoch+1}/{EPOCHS} - Perdida Train: {train_loss:.4f} - Perdida Val: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("   [!] Early stopping ejecutado.")
                break
    return model

print("Función generica de entrenamiento de los modelos listo.")

# %% [markdown]
# #### Función de evaluación de los modelos con las métricas

# %%
from sklearn.model_selection import StratifiedKFold

def cross_validate_model(model_class, model_name, X_train, y_train, vocab_size, embed_dim, num_classes, device, le, archivo_reporte):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    acc_list, prec_list, rec_list, f1_list = [], [], [], []
    print(f"\n--- Iniciando 5-Fold CV para {model_name} en Entrenamiento Efectivo ---")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_fold_train = torch.tensor(X_train[train_idx], dtype=torch.long)
        y_fold_train = torch.tensor(y_train[train_idx], dtype=torch.long)
        X_fold_val = torch.tensor(X_train[val_idx], dtype=torch.long)
        y_fold_val = torch.tensor(y_train[val_idx], dtype=torch.long)
        
        train_dataset = TensorDataset(X_fold_train, y_fold_train)
        val_dataset = TensorDataset(X_fold_val, y_fold_val)
        
        train_loader_fold = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader_fold = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        model = model_class(vocab_size, embed_dim, num_classes).to(device)
        print(f" Fold {fold+1}/5...")
        model = train_model(model, train_loader_fold, val_loader_fold, device)
        
        model.eval()
        preds_list = []
        labels_list = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader_fold:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                _, preds = torch.max(outputs, 1)
                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(y_batch.numpy())
                
        acc = accuracy_score(labels_list, preds_list)
        prec = precision_score(labels_list, preds_list, average='macro', zero_division=0)
        rec = recall_score(labels_list, preds_list, average='macro', zero_division=0)
        f1 = f1_score(labels_list, preds_list, average='macro', zero_division=0)
        
        acc_list.append(acc)
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        
    print(f"\n Resultados CV (5 Folds) para {model_name}:")
    print(f"Accuracy media: {np.mean(acc_list):.4f}")
    print(f"Precision media: {np.mean(prec_list):.4f}")
    print(f"Recall media: {np.mean(rec_list):.4f}")
    print(f"F1-Score media: {np.mean(f1_list):.4f}")
    
    with open(archivo_reporte, 'a', encoding='utf-8') as f:
        f.write(f"## {model_name}\n\n")
        f.write(f"### Resultados de Validación Cruzada (5-Folds en Entrenamiento Efectivo)\n")
        f.write(f"- **Accuracy Media:** {np.mean(acc_list):.4f}\n")
        f.write(f"- **Precision Media (macro):** {np.mean(prec_list):.4f}\n")
        f.write(f"- **Recall Media (macro):** {np.mean(rec_list):.4f}\n")
        f.write(f"- **F1-Score Media (macro):** {np.mean(f1_list):.4f}\n\n")


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_and_plot_conjuntos(model, nombre_modelo, train_loader, val_loader, test_loader, device, le, archivo_reporte):
    model.eval()
    
    conjuntos = [
        ('Entrenamiento Efectivo', train_loader),
        ('Validación', val_loader),
        ('Prueba', test_loader)
    ]
    
    resultados = {}
    
    with torch.no_grad():
        for nombre, loader in conjuntos:
            preds_list = []
            labels_list = []
            probs_list = []
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(y_batch.numpy())
                probs_list.extend(probs.cpu().numpy())
                
            resultados[nombre] = {
                'preds': np.array(preds_list),
                'labels': np.array(labels_list),
                'probs': np.array(probs_list)
            }
            
    metricas = {}
    for nombre in resultados.keys():
        lbls = resultados[nombre]['labels']
        prds = resultados[nombre]['preds']
        acc = accuracy_score(lbls, prds)
        prec = precision_score(lbls, prds, average='macro', zero_division=0)
        rec = recall_score(lbls, prds, average='macro', zero_division=0)
        f1 = f1_score(lbls, prds, average='macro', zero_division=0)
        
        y_labels = le.inverse_transform(lbls)
        p_labels = le.inverse_transform(prds)
        reporte_str = classification_report(y_labels, p_labels, labels=le.classes_, zero_division=0)
        
        metricas[nombre] = {
            'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1, 'reporte': reporte_str
        }
        
    fig_cm, axes_cm = plt.subplots(1, 3, figsize=(20, 6))
    fig_cm.suptitle(f'Matrices de Confusión - {nombre_modelo}', fontsize=16)
    
    for i, nombre in enumerate(resultados.keys()):
        cm = confusion_matrix(resultados[nombre]['labels'], resultados[nombre]['preds'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, ax=axes_cm[i])
        axes_cm[i].set_title(nombre)
        axes_cm[i].set_xlabel('Predicción')
        axes_cm[i].set_ylabel('Real')
        
    plt.tight_layout()
    nombre_base = nombre_modelo.lower().replace(' ', '_').replace('á', 'a')
    ruta_cm = os.path.join(RESULTADOS_DIR, f'cm_conjunta_{nombre_base}.png')
    plt.savefig(ruta_cm, dpi=300)
    plt.close()
    
    n_classes = len(le.classes_)
    fig_roc, axes_roc = plt.subplots(1, 3, figsize=(20, 6))
    fig_roc.suptitle(f'Curvas ROC Multiclase (OvR) - {nombre_modelo}', fontsize=16)
    
    auc_dict = {}
    
    for i, nombre in enumerate(resultados.keys()):
        y_bin = label_binarize(resultados[nombre]['labels'], classes=range(n_classes))
        probs = resultados[nombre]['probs']
        
        fpr, tpr, roc_auc = {}, {}, {}
        for c in range(n_classes):
            fpr[c], tpr[c], _ = roc_curve(y_bin[:, c], probs[:, c])
            roc_auc[c] = auc(fpr[c], tpr[c])
            axes_roc[i].plot(fpr[c], tpr[c], lw=2, label=f'{le.classes_[c]} (AUC = {roc_auc[c]:.3f})')
            
        all_fpr = np.unique(np.concatenate([fpr[c] for c in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for c in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[c], tpr[c])
        mean_tpr /= n_classes
        
        macro_auc = auc(all_fpr, mean_tpr)
        axes_roc[i].plot(all_fpr, mean_tpr, color='black', linestyle='--', lw=2.5, label=f'Macro-avg (AUC = {macro_auc:.3f})')
        axes_roc[i].plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5)
        axes_roc[i].set_xlim([0.0, 1.0])
        axes_roc[i].set_ylim([0.0, 1.05])
        axes_roc[i].set_xlabel('FPR')
        axes_roc[i].set_ylabel('TPR')
        axes_roc[i].set_title(nombre)
        axes_roc[i].legend(loc="lower right", fontsize=8)
        
        auc_dict[nombre] = {'macro': macro_auc, 'clases': roc_auc}
        
    plt.tight_layout()
    ruta_roc = os.path.join(RESULTADOS_DIR, f'roc_conjunta_{nombre_base}.png')
    plt.savefig(ruta_roc, dpi=300)
    plt.close()
    
    with open(archivo_reporte, 'a', encoding='utf-8') as f:
        f.write(f"## {nombre_modelo}\n\n")
        
        for nombre in resultados.keys():
            f.write(f"### Resultados en {nombre}\n")
            f.write(f"- **Accuracy:** {metricas[nombre]['acc']:.4f}\n")
            f.write(f"- **Precision (macro):** {metricas[nombre]['prec']:.4f}\n")
            f.write(f"- **Recall (macro):** {metricas[nombre]['rec']:.4f}\n")
            f.write(f"- **F1-Score (macro):** {metricas[nombre]['f1']:.4f}\n\n")
            f.write("```text\n")
            f.write(metricas[nombre]['reporte'])
            f.write("\n```\n\n")
            
            f.write(f"**AUC ROC ({nombre}):**\n")
            f.write(f"- Macro-average: {auc_dict[nombre]['macro']:.4f}\n")
            for c in range(n_classes):
                f.write(f"- {le.classes_[c]}: {auc_dict[nombre]['clases'][c]:.4f}\n")
            f.write("\n")
            
        f.write("### Gráficas de Evaluación Conjuntas\n\n")
        f.write(f"**Matrices de Confusión:**\n\n![Matrices de Confusión {nombre_modelo}](./cm_conjunta_{nombre_base}.png)\n\n")
        f.write(f"**Curvas ROC:**\n\n![Curvas ROC {nombre_modelo}](./roc_conjunta_{nombre_base}.png)\n\n")
        f.write("---\n\n")

    print(f"Evaluación conjunta para {nombre_modelo} finalizada y reporte actualizado.")


# %% [markdown]
# ### 5. Ejecución del pipeline

# %%
# Inicializar el archivo de reporte
archivo_reporte = os.path.join(RESULTADOS_DIR, 'reporte_redes_neuronales.md')
with open(archivo_reporte, 'w', encoding='utf-8') as f:
    f.write("# Reporte de Evaluación de Redes Neuronales\n\n")
    f.write("Resultados de métricas de rendimiento para los modelos CNN y LSTM.\n\n")

# CNN
print("Evaluación de CNN")
# Cross Validation
cross_validate_model(CNNModel, "CNN", X_train, y_train, vocab_size, EMBEDDING_DIM, num_classes, device, le, archivo_reporte)

# Entrenamiento Final
print("\nEntrenando modelo CNN Final")
model_cnn = CNNModel(vocab_size, EMBEDDING_DIM, num_classes).to(device)
model_cnn = train_model(model_cnn, train_loader, val_loader, device)

# Gráficas conjuntas (train, val, test)
print("\nEvaluando CNN Final en los 3 conjuntos estáticos...")
evaluate_and_plot_conjuntos(model_cnn, "CNN", train_loader, val_loader, test_loader, device, le, archivo_reporte)

# LSTM
print("\nEvaluación de LSTM")
# Cross Validation
cross_validate_model(LSTMModel, "LSTM", X_train, y_train, vocab_size, EMBEDDING_DIM, num_classes, device, le, archivo_reporte)

# Entrenamiento Final
print("\nEntrenando modelo LSTM Final")
model_lstm = LSTMModel(vocab_size, EMBEDDING_DIM, num_classes).to(device)
model_lstm = train_model(model_lstm, train_loader, val_loader, device)

# Gráficas conjuntas (train, val, test)
print("\nEvaluando LSTM Final en los 3 conjuntos estáticos...")
evaluate_and_plot_conjuntos(model_lstm, "LSTM", train_loader, val_loader, test_loader, device, le, archivo_reporte)

print(f"\nReporte y gráficas generadas y guardadas en: {RESULTADOS_DIR}")



