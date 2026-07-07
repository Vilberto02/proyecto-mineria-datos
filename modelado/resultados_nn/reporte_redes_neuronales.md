# Reporte de Evaluación de Redes Neuronales

Resultados de métricas de rendimiento para los modelos CNN y LSTM.

## CNN

### Resultados en Entrenamiento Efectivo
- **Accuracy:** 0.9886
- **Precision (macro):** 0.9886
- **Recall (macro):** 0.9888
- **F1-Score (macro):** 0.9887

```text
              precision    recall  f1-score   support

     Alegria       0.99      0.98      0.99      1304
       Miedo       0.99      0.99      0.99      1122
    Sorpresa       0.98      1.00      0.99      1147
    Tristeza       0.99      0.99      0.99      1503

    accuracy                           0.99      5076
   macro avg       0.99      0.99      0.99      5076
weighted avg       0.99      0.99      0.99      5076

```

**AUC ROC (Entrenamiento Efectivo):**
- Macro-average: 0.9991
- Alegria: 0.9988
- Miedo: 0.9996
- Sorpresa: 0.9993
- Tristeza: 0.9987

### Resultados en Validación
- **Accuracy:** 0.5850
- **Precision (macro):** 0.5935
- **Recall (macro):** 0.5801
- **F1-Score (macro):** 0.5843

```text
              precision    recall  f1-score   support

     Alegria       0.55      0.53      0.54       559
       Miedo       0.76      0.65      0.70       481
    Sorpresa       0.50      0.46      0.48       492
    Tristeza       0.57      0.68      0.62       644

    accuracy                           0.59      2176
   macro avg       0.59      0.58      0.58      2176
weighted avg       0.59      0.59      0.58      2176

```

**AUC ROC (Validación):**
- Macro-average: 0.8074
- Alegria: 0.7716
- Miedo: 0.8735
- Sorpresa: 0.7779
- Tristeza: 0.8055

### Resultados en Prueba
- **Accuracy:** 0.5720
- **Precision (macro):** 0.5899
- **Recall (macro):** 0.5689
- **F1-Score (macro):** 0.5755

```text
              precision    recall  f1-score   support

     Alegria       0.57      0.50      0.53       466
       Miedo       0.78      0.64      0.71       401
    Sorpresa       0.49      0.48      0.49       409
    Tristeza       0.52      0.65      0.58       537

    accuracy                           0.57      1813
   macro avg       0.59      0.57      0.58      1813
weighted avg       0.58      0.57      0.57      1813

```

**AUC ROC (Prueba):**
- Macro-average: 0.8024
- Alegria: 0.7768
- Miedo: 0.8778
- Sorpresa: 0.7806
- Tristeza: 0.7731

### Gráficas de Evaluación Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión CNN](./cm_conjunta_cnn.png)

**Curvas ROC:**

![Curvas ROC CNN](./roc_conjunta_cnn.png)

---

## LSTM

### Resultados en Entrenamiento Efectivo
- **Accuracy:** 0.8132
- **Precision (macro):** 0.8142
- **Recall (macro):** 0.8117
- **F1-Score (macro):** 0.8115

```text
              precision    recall  f1-score   support

     Alegria       0.83      0.79      0.81      1304
       Miedo       0.84      0.76      0.80      1122
    Sorpresa       0.76      0.86      0.81      1147
    Tristeza       0.83      0.84      0.83      1503

    accuracy                           0.81      5076
   macro avg       0.81      0.81      0.81      5076
weighted avg       0.82      0.81      0.81      5076

```

**AUC ROC (Entrenamiento Efectivo):**
- Macro-average: 0.9506
- Alegria: 0.9459
- Miedo: 0.9478
- Sorpresa: 0.9546
- Tristeza: 0.9537

### Resultados en Validación
- **Accuracy:** 0.5492
- **Precision (macro):** 0.5523
- **Recall (macro):** 0.5470
- **F1-Score (macro):** 0.5487

```text
              precision    recall  f1-score   support

     Alegria       0.54      0.51      0.52       559
       Miedo       0.64      0.57      0.60       481
    Sorpresa       0.46      0.51      0.49       492
    Tristeza       0.57      0.60      0.58       644

    accuracy                           0.55      2176
   macro avg       0.55      0.55      0.55      2176
weighted avg       0.55      0.55      0.55      2176

```

**AUC ROC (Validación):**
- Macro-average: 0.7848
- Alegria: 0.7686
- Miedo: 0.8350
- Sorpresa: 0.7631
- Tristeza: 0.7713

### Resultados en Prueba
- **Accuracy:** 0.5378
- **Precision (macro):** 0.5442
- **Recall (macro):** 0.5380
- **F1-Score (macro):** 0.5401

```text
              precision    recall  f1-score   support

     Alegria       0.52      0.50      0.51       466
       Miedo       0.66      0.59      0.62       401
    Sorpresa       0.45      0.51      0.48       409
    Tristeza       0.55      0.55      0.55       537

    accuracy                           0.54      1813
   macro avg       0.54      0.54      0.54      1813
weighted avg       0.54      0.54      0.54      1813

```

**AUC ROC (Prueba):**
- Macro-average: 0.7713
- Alegria: 0.7418
- Miedo: 0.8245
- Sorpresa: 0.7576
- Tristeza: 0.7601

### Gráficas de Evaluación Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión LSTM](./cm_conjunta_lstm.png)

**Curvas ROC:**

![Curvas ROC LSTM](./roc_conjunta_lstm.png)

---

