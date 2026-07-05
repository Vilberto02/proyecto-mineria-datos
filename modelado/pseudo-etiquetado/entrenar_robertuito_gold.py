import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

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
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1 = f1_score(labels, predictions, average='macro')
    acc = accuracy_score(labels, predictions)
    return {'f1_macro': f1, 'accuracy': acc}

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if self.class_weights is not None:
            # Move class_weights to the same device as logits
            class_weights = self.class_weights.to(logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
        return (loss, outputs) if return_outputs else loss

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.abspath(os.path.join(current_dir, '..', 'datos'))
    if not os.path.exists(base_path):
        base_path = os.path.join(current_dir, 'datos') # fallback for colab if flat

    gold_set_path = os.path.join(base_path, 'gold_set.csv')
    
    print("Cargando Gold Set...")
    df = pd.read_csv(gold_set_path, sep=';')
    
    # Preparar labels
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['emocion'])
    
    # Guardar los nombres de las clases para luego
    classes = le.classes_
    num_labels = len(classes)
    np.save(os.path.join(base_path, 'classes.npy'), classes)
    print(f"Clases encontradas: {classes}")
    
    # Calcular class weights
    class_counts = df['label_encoded'].value_counts().sort_index().values
    total_samples = len(df)
    # Inverse frequency weights
    weights = total_samples / (num_labels * class_counts)
    class_weights = torch.tensor(weights, dtype=torch.float)
    print(f"Pesos de clase calculados: {class_weights}")
    
    # Modelo pre-entrenado
    model_name = "pysentimiento/robertuito-base-uncased"
    print(f"Cargando tokenizador y modelo: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    
    # Preparar Dataset
    # Usaremos todo el gold set para entrenamiento, idealmente se debe hacer CV
    train_dataset = EmotionDataset(df['content'].values, df['label_encoded'].values, tokenizer)
    
    results_path = os.path.abspath(os.path.join(current_dir, '..', 'resultados_transformers'))
    if not os.path.exists(os.path.dirname(results_path)):
        # If running flat in colab
        results_path = os.path.join(current_dir, 'resultados_transformers')
        
    training_args = TrainingArguments(
        output_dir=os.path.join(results_path, 'robertuito_gold'),
        num_train_epochs=4,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_strategy='epoch'
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        class_weights=class_weights,
        compute_metrics=compute_metrics
    )
    
    print("Iniciando entrenamiento...")
    trainer.train()
    
    # Guardar modelo final
    model_path = os.path.join(results_path, 'robertuito_gold_final')
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    main()
