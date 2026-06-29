# Reporte de evaluación de modelos Transformer

Resultados de métricas de rendimiento para los modelos BETO (BERT en Español) y RoBERTa (RoBERTuito).

---

## Comparativa Global

| Métrica           | BETO   | RoBERTa |
| ----------------- | ------ | ------- |
| Accuracy          | 0.7253 | 0.7248  |
| Precision (macro) | 0.7369 | 0.7245  |
| Recall (macro)    | 0.7244 | 0.7235  |
| F1-Score (macro)  | 0.7246 | 0.7235  |
| AUC (macro)       | 0.9097 | 0.9074  |

---

## BETO

### Resultados en Entrenamiento Efectivo (~56%)

- **Accuracy:** 0.8930
- **Precision (macro):** 0.8964
- **Recall (macro):** 0.8937
- **F1-Score (macro):** 0.8929

### Resultados en Validación (~24%)

- **Accuracy:** 0.7247
- **Precision (macro):** 0.7313
- **Recall (macro):** 0.7228
- **F1-Score (macro):** 0.7225

### Resultados en Prueba y Evaluación (20%)

- **Accuracy:** 0.7253
- **Precision (macro):** 0.7369
- **Recall (macro):** 0.7244
- **F1-Score (macro):** 0.7246

### AUC (Área Bajo la Curva ROC)

- **Alegria:** 0.8812
- **Miedo:** 0.9432
- **Sorpresa:** 0.8990
- **Tristeza:** 0.9140
- **Macro-promedio:** 0.9097

### Reporte por Clase

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.7872    | 0.5794 | 0.6675   | 466     |
| Miedo    | 0.8333    | 0.7855 | 0.8087   | 401     |
| Sorpresa | 0.6253    | 0.7262 | 0.6719   | 409     |
| Tristeza | 0.7018    | 0.8063 | 0.7504   | 537     |

### Matriz de Confusión

![Matriz de Confusión BETO](/content/resultados_transformers/cm_beto.png)

### Curva ROC

![Curva ROC BETO](/content/resultados_transformers/roc_beto.png)

### Classification Report

```text
              precision    recall  f1-score   support

     Alegria       0.79      0.58      0.67       466
       Miedo       0.83      0.79      0.81       401
    Sorpresa       0.63      0.73      0.67       409
    Tristeza       0.70      0.81      0.75       537

    accuracy                           0.73      1813
   macro avg       0.74      0.72      0.72      1813
weighted avg       0.74      0.73      0.72      1813
```

---

## RoBERTa

### Resultados en Entrenamiento Efectivo (~56%)

- **Accuracy:** 0.8262
- **Precision (macro):** 0.8263
- **Recall (macro):** 0.8259
- **F1-Score (macro):** 0.8258

### Resultados en Validación (~24%)

- **Accuracy:** 0.7206
- **Precision (macro):** 0.7194
- **Recall (macro):** 0.7185
- **F1-Score (macro):** 0.7186

### Resultados en Prueba y Evaluación (20%)

- **Accuracy:** 0.7248
- **Precision (macro):** 0.7245
- **Recall (macro):** 0.7235
- **F1-Score (macro):** 0.7235

### AUC (Área Bajo la Curva ROC)

- **Alegria:** 0.8806
- **Miedo:** 0.9432
- **Sorpresa:** 0.8984
- **Tristeza:** 0.9063
- **Macro-promedio:** 0.9074

### Reporte por Clase

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.7319    | 0.6738 | 0.7017   | 466     |
| Miedo    | 0.7897    | 0.8055 | 0.7975   | 401     |
| Sorpresa | 0.6519    | 0.6455 | 0.6486   | 409     |
| Tristeza | 0.7246    | 0.7691 | 0.7462   | 537     |

### Matriz de Confusión

![Matriz de Confusión RoBERTa](/content/resultados_transformers/cm_roberta.png)

### Curva ROC

![Curva ROC RoBERTa](/content/resultados_transformers/roc_roberta.png)

### Classification Report

```text
              precision    recall  f1-score   support

     Alegria       0.73      0.67      0.70       466
       Miedo       0.79      0.81      0.80       401
    Sorpresa       0.65      0.65      0.65       409
    Tristeza       0.72      0.77      0.75       537

    accuracy                           0.72      1813
   macro avg       0.72      0.72      0.72      1813
weighted avg       0.72      0.72      0.72      1813
```
