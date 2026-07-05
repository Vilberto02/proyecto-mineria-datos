import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# pyrefly: ignore [missing-import]
from torch.utils.data import DataLoader, Dataset
# pyrefly: ignore [missing-import]
import hdbscan
import os
from sklearn.preprocessing import LabelEncoder

class InferenceDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }

def get_predictions_and_embeddings(model, dataloader, device):
    model.eval()
    predictions = []
    embeddings = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            logits = outputs.logits
            
            # Predict
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            
            # Get embeddings from the last hidden state (CLS token is at index 0)
            hidden_states = outputs.hidden_states[-1]
            cls_embeddings = hidden_states[:, 0, :].cpu().numpy()
            embeddings.extend(cls_embeddings)
            
    return np.array(predictions), np.array(embeddings)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(current_dir, '..', 'datos'))
    results_path = os.path.abspath(os.path.join(current_dir, '..', 'resultados_transformers'))
    
    if not os.path.exists(base_path):
        base_path = os.path.join(current_dir, 'datos')
        results_path = os.path.join(current_dir, 'resultados_transformers')
        
    silver_path = os.path.join(base_path, 'silver_target_dataset.csv')
    model_path = os.path.join(results_path, 'robertuito_gold_final')
    classes_path = os.path.join(base_path, 'classes.npy')
    
    print("Cargando dataset Silver...")
    df_silver = pd.read_csv(silver_path, sep=';')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: El modelo no se encuentra en {model_path}. Debes entrenarlo primero.")
        return
        
    print("Cargando modelo y tokenizador...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    
    classes = np.load(classes_path, allow_pickle=True)
    
    print("Generando predicciones y embeddings (puede tomar un momento)...")
    dataset = InferenceDataset(df_silver['content'].values, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    predictions, embeddings = get_predictions_and_embeddings(model, dataloader, device)
    
    # Asignar predicciones (Silver Labels)
    df_silver['pred_encoded'] = predictions
    df_silver['silver_label'] = [classes[p] for p in predictions]
    
    print("Aplicando HDBSCAN...")
    # Para datasets grandes o embeddings de alta dimensionalidad (768), puede ser util aplicar UMAP primero
    # Por simplicidad directa aplicamos HDBSCAN, aunque puede requerir ajuste de parametros
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings)
    df_silver['cluster_id'] = cluster_labels
    
    print("Construyendo tabla de contingencia y limpiando etiquetas...")
    # Identificar etiqueta dominante por cluster
    cluster_dominant_labels = {}
    for cluster in df_silver['cluster_id'].unique():
        if cluster == -1:
            continue # Ruido segun HDBSCAN
        cluster_data = df_silver[df_silver['cluster_id'] == cluster]
        dominant_label = cluster_data['silver_label'].mode()[0]
        cluster_dominant_labels[cluster] = dominant_label
        
    # Aplicar regla de descarte automático
    clean_indices = []
    for idx, row in df_silver.iterrows():
        c_id = row['cluster_id']
        if c_id == -1:
            # Si el cluster es ruido, descartamos
            continue
            
        dominant = cluster_dominant_labels[c_id]
        if row['silver_label'] == dominant:
            clean_indices.append(idx)
            
    df_clean_silver = df_silver.loc[clean_indices].copy()
    print(f"Total registros Silver originales: {len(df_silver)}")
    print(f"Registros Silver validados y conservados: {len(df_clean_silver)}")
    print(f"Descartados por conflicto: {len(df_silver) - len(df_clean_silver)}")
    
    # Unir con el Gold Set para formar el Clean Dataset final
    gold_set_path = os.path.join(base_path, 'gold_set.csv')
    df_gold = pd.read_csv(gold_set_path, sep=';')
    
    # Usar la etiqueta final
    df_gold['final_emocion'] = df_gold['emocion']
    df_clean_silver['final_emocion'] = df_clean_silver['silver_label']
    
    df_final = pd.concat([df_gold, df_clean_silver], ignore_index=True)
    
    # Guardar
    output_path = os.path.join(base_path, 'dataset_limpio_final.csv')
    df_final.to_csv(output_path, sep=';', index=False)
    print(f"Dataset limpio final guardado en {output_path} con un total de {len(df_final)} registros.")

if __name__ == "__main__":
    main()
