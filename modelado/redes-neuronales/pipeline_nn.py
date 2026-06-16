import os
import sys
import collections
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
import torch.nn as nn
# pyrefly: ignore [missing-import]
import torch.optim as optim
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, TensorDataset

# Configuración de carpetas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)
from app import cargar_datasets

DATOS_DIR = os.path.join(BASE_DIR, 'datos')
RESULTADOS_DIR = os.path.join(BASE_DIR, 'resultados')
os.makedirs(RESULTADOS_DIR, exist_ok=True)

# Hiperparámetros
MAX_WORDS = 10000
MAX_LEN = 100
EMBEDDING_DIM = 64
BATCH_SIZE = 32
EPOCHS = 10
PATIENCE = 3 # Early stopping
LEARNING_RATE = 0.001

# Tokenización
def build_vocab(texts, max_words=MAX_WORDS):
    words = []
    for text in texts:
        words.extend(str(text).split())
    counter = collections.Counter(words)
    # 0 es PAD, 1 es UNK
    vocab = {word: idx + 2 for idx, (word, _) in enumerate(counter.most_common(max_words - 2))}
    return vocab

# Secuencias de texto
def texts_to_sequences(texts, vocab):
    seqs = []
    for text in texts:
        seq = [vocab.get(w, 1) for w in str(text).split()]
        seqs.append(seq)
    return seqs

# Padding
def pad_sequences(seqs, maxlen=MAX_LEN):
    padded = np.zeros((len(seqs), maxlen), dtype=int)
    for i, seq in enumerate(seqs):
        if len(seq) > maxlen:
            padded[i, :] = seq[:maxlen]
        else:
            padded[i, -len(seq):] = seq
    return padded

# Modelos PyTorch
# CNN
class CNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=5)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.embedding(x) # (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1) # (batch, embed_dim, seq_len)
        x = self.conv(x)
        x = self.relu(x)
        x = torch.max(x, dim=2)[0] # GlobalMaxPooling1D
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

# Entrenamiento
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
        print(f"   Época {epoch+1}/{EPOCHS} - Pérdida Entrenamiento: {train_loss:.4f} - Pérdida Prueba: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("   [!] Early stopping alcanzado.")
                break
    return model

# Evaluación
def evaluate_and_report(model, nombre_modelo, dataloader, device, le, archivo_reporte):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            
    # Generar Matriz de Confusión
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.tight_layout()
    
    nombre_archivo_cm = f'cm_nn_{nombre_modelo.lower()}.png'
    ruta_cm = os.path.join(RESULTADOS_DIR, nombre_archivo_cm)
    plt.savefig(ruta_cm, dpi=300)
    plt.close()
    
    # Escribir resultados
    with open(archivo_reporte, 'a', encoding='utf-8') as f:
        f.write(f"## {nombre_modelo}\n\n")
        f.write(f"- **Accuracy:** {accuracy_score(all_labels, all_preds):.4f}\n")
        f.write(f"- **Precision (macro):** {precision_score(all_labels, all_preds, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **Recall (macro):** {recall_score(all_labels, all_preds, average='macro', zero_division=0):.4f}\n")
        f.write(f"- **F1-Score (macro):** {f1_score(all_labels, all_preds, average='macro', zero_division=0):.4f}\n\n")
        
        f.write("```text\n")
        # Inversión de labels para el reporte de clasificación
        y_test_labels = le.inverse_transform(all_labels)
        y_pred_labels = le.inverse_transform(all_preds)
        f.write(classification_report(y_test_labels, y_pred_labels, labels=le.classes_, zero_division=0))
        f.write("\n```\n\n")
        
        ruta_cm_md = ruta_cm.replace('\\', '/')
        f.write(f"**Matriz de Confusión:**\n\n![Matriz de Confusión {nombre_modelo}](file:///{ruta_cm_md})\n\n")


def ejecutar_pipeline_nn():
    print("1. Carga de datos...")
    df = cargar_datasets()
    
    # Preprocesamiento Nulos
    df = df.dropna(subset=['lemas', 'emocion'])
    
    # Codificar Emociones
    print("2. Codificacion de emociones y preparacion de texto...")
    le = LabelEncoder()
    y = le.fit_transform(df['emocion'])
    texts = df['lemas'].astype(str).tolist()
    
    # NLP
    vocab = build_vocab(texts)
    vocab_size = min(MAX_WORDS, len(vocab) + 2)
    seqs = texts_to_sequences(texts, vocab)
    X = pad_sequences(seqs)
    
    print("3. Division del dataset (70% entrenamiento, 30% prueba)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Tensores
    X_train_t = torch.tensor(X_train, dtype=torch.long)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.long)
    y_test_t = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Utilizando dispositivo: {device}")
    
    num_classes = len(le.classes_)
    modelos = {
        'CNN': CNNModel(vocab_size, EMBEDDING_DIM, num_classes).to(device),
        'LSTM': LSTMModel(vocab_size, EMBEDDING_DIM, num_classes).to(device)
    }
    
    archivo_reporte = os.path.join(RESULTADOS_DIR, 'reporte_redes_neuronales.md')
    with open(archivo_reporte, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Evaluación de Redes Neuronales Profundas\n\n")
        f.write("Resultados de las métricas de rendimiento para los modelos CNN y LSTM.\n\n")
        
    for nombre, modelo in modelos.items():
        print(f"\nEntrenando {nombre}")
        modelo = train_model(modelo, train_loader, test_loader, device)
        
        print(f"Evaluando {nombre} en conjunto de prueba")
        evaluate_and_report(modelo, nombre, test_loader, device, le, archivo_reporte)
        
    print(f"\nProceso completado. Reporte guardado en: {archivo_reporte}")

if __name__ == '__main__':
    ejecutar_pipeline_nn()
