# Reporte de Evaluación de Modelos Transformer

Resultados de métricas de rendimiento para los modelos BETO (BERT en Español) y RoBERTa (BERTIN - Español).

## Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Max length | 128 |
| Batch size | 16 |
| Learning rate | 2e-05 |
| Epochs | 3 |
| Weight decay | 0.01 |
| Warmup steps | 100 |
| Early stopping patience | 2 |
| Pesos de clase | balanced |

---

## Comparativa Global

| Métrica | BETO | RoBERTa |
|---------|------|---------|
| Accuracy | 0.7338 | 0.7287 |
| Precision (macro) | 0.7338 | 0.7318 |
| Recall (macro) | 0.7333 | 0.7305 |
| F1-Score (macro) | 0.7335 | 0.7294 |
| AUC (macro) | 0.9095 | 0.9054 |

---

## BETO

### Métricas Globales

- **Accuracy:** 0.7338
- **Precision (macro):** 0.7338
- **Recall (macro):** 0.7333
- **F1-Score (macro):** 0.7335

### AUC (Área Bajo la Curva ROC)

- **Alegria:** 0.8843
- **Miedo:** 0.9450
- **Sorpresa:** 0.9030
- **Tristeza:** 0.9049
- **Macro-promedio:** 0.9095

### Reporte por Clase

| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| Alegria | 0.7068 | 0.6967 | 0.7017 | 699 |
| Miedo | 0.8003 | 0.8070 | 0.8036 | 601 |
| Sorpresa | 0.6856 | 0.6667 | 0.6760 | 615 |
| Tristeza | 0.7424 | 0.7627 | 0.7525 | 805 |

### Matriz de Confusión

![Matriz de Confusión BETO](/content/resultados_transformers/cm_beto.png)

### Curva ROC

![Curva ROC BETO](/content/resultados_transformers/roc_beto.png)

### Classification Report

```text
              precision    recall  f1-score   support

     Alegria       0.71      0.70      0.70       699
       Miedo       0.80      0.81      0.80       601
    Sorpresa       0.69      0.67      0.68       615
    Tristeza       0.74      0.76      0.75       805

    accuracy                           0.73      2720
   macro avg       0.73      0.73      0.73      2720
weighted avg       0.73      0.73      0.73      2720
```

---

## RoBERTa

### Métricas Globales

- **Accuracy:** 0.7287
- **Precision (macro):** 0.7318
- **Recall (macro):** 0.7305
- **F1-Score (macro):** 0.7294

### AUC (Área Bajo la Curva ROC)

- **Alegria:** 0.8879
- **Miedo:** 0.9343
- **Sorpresa:** 0.8982
- **Tristeza:** 0.9002
- **Macro-promedio:** 0.9054

### Reporte por Clase

| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| Alegria | 0.7574 | 0.6567 | 0.7034 | 699 |
| Miedo | 0.7933 | 0.7920 | 0.7927 | 601 |
| Sorpresa | 0.6383 | 0.7317 | 0.6818 | 615 |
| Tristeza | 0.7379 | 0.7416 | 0.7398 | 805 |

### Matriz de Confusión

![Matriz de Confusión RoBERTa](/content/resultados_transformers/cm_roberta.png)

### Curva ROC

![Curva ROC RoBERTa](/content/resultados_transformers/roc_roberta.png)

### Classification Report

```text
              precision    recall  f1-score   support

     Alegria       0.76      0.66      0.70       699
       Miedo       0.79      0.79      0.79       601
    Sorpresa       0.64      0.73      0.68       615
    Tristeza       0.74      0.74      0.74       805

    accuracy                           0.73      2720
   macro avg       0.73      0.73      0.73      2720
weighted avg       0.73      0.73      0.73      2720
```

