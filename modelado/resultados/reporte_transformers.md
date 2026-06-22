# Reporte de Evaluación de Modelos Transformer

Resultados de métricas de rendimiento para los modelos BETO (BERT en Español) y RoBERTa (XLM-RoBERTa).

## Hiperparámetros

| Parámetro | Valor |
|-----------|-------|
| Max length | 128 |
| Batch size | 16 |
| Learning rate | 2e-05 |
| Epochs | 5 |
| Weight decay | 0.01 |
| Warmup steps | 100 |
| Early stopping patience | 2 |
| Pesos de clase | balanced |

---

## Comparativa Global

| Métrica | BETO | RoBERTa |
|---------|------|---------|
| Accuracy | 0.5752 | 0.5554 |
| Precision (macro) | 0.5054 | 0.4833 |
| Recall (macro) | 0.4787 | 0.4592 |
| F1-Score (macro) | 0.4817 | 0.4618 |
| AUC (macro) | 0.8222 | 0.8020 |

---

## BETO

### Métricas Globales

- **Accuracy:** 0.5752
- **Precision (macro):** 0.5054
- **Recall (macro):** 0.4787
- **F1-Score (macro):** 0.4817

### AUC (Área Bajo la Curva ROC)

- **Alegría:** 0.8246
- **Asco:** 0.7795
- **Ira:** 0.7857
- **Miedo:** 0.8932
- **Sorpresa:** 0.8112
- **Tristeza:** 0.8380
- **Macro-promedio:** 0.8222

### Reporte por Clase

| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| Alegría | 0.6121 | 0.6094 | 0.6108 | 699 |
| Asco | 0.3258 | 0.2407 | 0.2768 | 241 |
| Ira | 0.2807 | 0.0941 | 0.1410 | 170 |
| Miedo | 0.7354 | 0.6705 | 0.7015 | 601 |
| Sorpresa | 0.4887 | 0.6000 | 0.5387 | 615 |
| Tristeza | 0.5897 | 0.6571 | 0.6216 | 805 |

### Matriz de Confusión

![Matriz de Confusión BETO](/content/drive/MyDrive/MiDrive/proyecto-mineria-datos/modelado/resultados/cm_beto.png)

### Curva ROC

![Curva ROC BETO](/content/drive/MyDrive/MiDrive/proyecto-mineria-datos/modelado/resultados/roc_beto.png)

### Classification Report

```text
              precision    recall  f1-score   support

     Alegría       0.61      0.61      0.61       699
        Asco       0.33      0.24      0.28       241
         Ira       0.28      0.09      0.14       170
       Miedo       0.74      0.67      0.70       601
    Sorpresa       0.49      0.60      0.54       615
    Tristeza       0.59      0.66      0.62       805

    accuracy                           0.58      3131
   macro avg       0.51      0.48      0.48      3131
weighted avg       0.57      0.58      0.57      3131
```

---

## RoBERTa

### Métricas Globales

- **Accuracy:** 0.5554
- **Precision (macro):** 0.4833
- **Recall (macro):** 0.4592
- **F1-Score (macro):** 0.4618

### AUC (Área Bajo la Curva ROC)

- **Alegría:** 0.8030
- **Asco:** 0.7438
- **Ira:** 0.7540
- **Miedo:** 0.8883
- **Sorpresa:** 0.7910
- **Tristeza:** 0.8308
- **Macro-promedio:** 0.8020

### Reporte por Clase

| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| Alegría | 0.6111 | 0.5665 | 0.5880 | 699 |
| Asco | 0.1932 | 0.1660 | 0.1786 | 241 |
| Ira | 0.3226 | 0.1176 | 0.1724 | 170 |
| Miedo | 0.7325 | 0.7155 | 0.7239 | 601 |
| Sorpresa | 0.4391 | 0.5512 | 0.4888 | 615 |
| Tristeza | 0.6012 | 0.6385 | 0.6193 | 805 |

### Matriz de Confusión

![Matriz de Confusión RoBERTa](/content/drive/MyDrive/MiDrive/proyecto-mineria-datos/modelado/resultados/cm_roberta.png)

### Curva ROC

![Curva ROC RoBERTa](/content/drive/MyDrive/MiDrive/proyecto-mineria-datos/modelado/resultados/roc_roberta.png)

### Classification Report

```text
              precision    recall  f1-score   support

     Alegría       0.61      0.57      0.59       699
        Asco       0.19      0.17      0.18       241
         Ira       0.32      0.12      0.17       170
       Miedo       0.73      0.72      0.72       601
    Sorpresa       0.44      0.55      0.49       615
    Tristeza       0.60      0.64      0.62       805

    accuracy                           0.56      3131
   macro avg       0.48      0.46      0.46      3131
weighted avg       0.55      0.56      0.55      3131
```

