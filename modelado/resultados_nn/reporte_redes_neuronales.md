# Reporte de Evaluación de Redes Neuronales

Resultados de métricas de rendimiento para los modelos CNN y LSTM.

## CNN

### Resultados en Entrenamiento Efectivo
- **Accuracy:** 0.9978
- **Precision (macro):** 0.9987
- **Recall (macro):** 0.9978
- **F1-Score (macro):** 0.9983

```text
              precision    recall  f1-score   support

     Alegria       1.00      1.00      1.00       262
       Miedo       0.99      1.00      1.00       387
    Sorpresa       1.00      1.00      1.00        64
    Tristeza       1.00      1.00      1.00       204

    accuracy                           1.00       917
   macro avg       1.00      1.00      1.00       917
weighted avg       1.00      1.00      1.00       917

```

**AUC ROC (Entrenamiento Efectivo):**
- Macro-average: 1.0000
- Alegria: 1.0000
- Miedo: 1.0000
- Sorpresa: 1.0000
- Tristeza: 1.0000

### Resultados en Validación
- **Accuracy:** 0.7107
- **Precision (macro):** 0.6992
- **Recall (macro):** 0.5600
- **F1-Score (macro):** 0.5668

```text
              precision    recall  f1-score   support

     Alegria       0.70      0.76      0.73       112
       Miedo       0.76      0.90      0.83       167
    Sorpresa       0.75      0.11      0.19        27
    Tristeza       0.59      0.47      0.52        88

    accuracy                           0.71       394
   macro avg       0.70      0.56      0.57       394
weighted avg       0.70      0.71      0.69       394

```

**AUC ROC (Validación):**
- Macro-average: 0.8502
- Alegria: 0.8818
- Miedo: 0.9111
- Sorpresa: 0.7634
- Tristeza: 0.8378

### Resultados en Prueba
- **Accuracy:** 0.7317
- **Precision (macro):** 0.5307
- **Recall (macro):** 0.5578
- **F1-Score (macro):** 0.5414

```text
              precision    recall  f1-score   support

     Alegria       0.78      0.82      0.80        93
       Miedo       0.77      0.92      0.84       139
    Sorpresa       0.00      0.00      0.00        23
    Tristeza       0.58      0.49      0.53        73

    accuracy                           0.73       328
   macro avg       0.53      0.56      0.54       328
weighted avg       0.67      0.73      0.70       328

```

**AUC ROC (Prueba):**
- Macro-average: 0.8741
- Alegria: 0.9322
- Miedo: 0.9407
- Sorpresa: 0.7960
- Tristeza: 0.8201

### Gráficas de Evaluación Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión CNN](./cm_conjunta_cnn.png)

**Curvas ROC:**

![Curvas ROC CNN](./roc_conjunta_cnn.png)

---

## LSTM

### Resultados en Entrenamiento Efectivo
- **Accuracy:** 0.8811
- **Precision (macro):** 0.8764
- **Recall (macro):** 0.7384
- **F1-Score (macro):** 0.7349

```text
              precision    recall  f1-score   support

     Alegria       0.91      0.95      0.93       262
       Miedo       0.96      0.93      0.95       387
    Sorpresa       0.90      0.14      0.24        64
    Tristeza       0.73      0.94      0.82       204

    accuracy                           0.88       917
   macro avg       0.88      0.74      0.73       917
weighted avg       0.89      0.88      0.86       917

```

**AUC ROC (Entrenamiento Efectivo):**
- Macro-average: 0.9839
- Alegria: 0.9904
- Miedo: 0.9900
- Sorpresa: 0.9739
- Tristeza: 0.9802

### Resultados en Validación
- **Accuracy:** 0.6294
- **Precision (macro):** 0.4670
- **Recall (macro):** 0.5053
- **F1-Score (macro):** 0.4810

```text
              precision    recall  f1-score   support

     Alegria       0.67      0.71      0.69       112
       Miedo       0.76      0.67      0.71       167
    Sorpresa       0.00      0.00      0.00        27
    Tristeza       0.44      0.64      0.52        88

    accuracy                           0.63       394
   macro avg       0.47      0.51      0.48       394
weighted avg       0.61      0.63      0.61       394

```

**AUC ROC (Validación):**
- Macro-average: 0.8059
- Alegria: 0.8468
- Miedo: 0.8359
- Sorpresa: 0.7470
- Tristeza: 0.7878

### Resultados en Prueba
- **Accuracy:** 0.6646
- **Precision (macro):** 0.4853
- **Recall (macro):** 0.5246
- **F1-Score (macro):** 0.5026

```text
              precision    recall  f1-score   support

     Alegria       0.65      0.80      0.72        93
       Miedo       0.79      0.74      0.76       139
    Sorpresa       0.00      0.00      0.00        23
    Tristeza       0.50      0.56      0.53        73

    accuracy                           0.66       328
   macro avg       0.49      0.52      0.50       328
weighted avg       0.63      0.66      0.64       328

```

**AUC ROC (Prueba):**
- Macro-average: 0.8376
- Alegria: 0.8930
- Miedo: 0.8796
- Sorpresa: 0.7728
- Tristeza: 0.7974

### Gráficas de Evaluación Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión LSTM](./cm_conjunta_lstm.png)

**Curvas ROC:**

![Curvas ROC LSTM](./roc_conjunta_lstm.png)

---

