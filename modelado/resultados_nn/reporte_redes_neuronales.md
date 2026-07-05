# Reporte de Evaluación de Redes Neuronales

Resultados de métricas de rendimiento para los modelos CNN y LSTM.

## CNN

### Resultados de Validación Cruzada (5-Folds en Entrenamiento Efectivo)
- **Accuracy Media:** 0.6772
- **Precision Media (macro):** 0.6200
- **Recall Media (macro):** 0.5205
- **F1-Score Media (macro):** 0.5172

## CNN

### Resultados en Entrenamiento Efectivo
- **Accuracy:** 0.9989
- **Precision (macro):** 0.9990
- **Recall (macro):** 0.9994
- **F1-Score (macro):** 0.9992

```text
              precision    recall  f1-score   support

     Alegria       1.00      1.00      1.00       262
       Miedo       1.00      1.00      1.00       387
    Sorpresa       1.00      1.00      1.00        64
    Tristeza       1.00      1.00      1.00       204

    accuracy                           1.00       917
   macro avg       1.00      1.00      1.00       917
weighted avg       1.00      1.00      1.00       917

```

**AUC ROC (Entrenamiento Efectivo):**
- Macro-average: 1.0000
- Alegria: 1.0000
- Miedo: 0.9999
- Sorpresa: 1.0000
- Tristeza: 1.0000

### Resultados en Validación
- **Accuracy:** 0.7183
- **Precision (macro):** 0.6899
- **Recall (macro):** 0.5702
- **F1-Score (macro):** 0.5699

```text
              precision    recall  f1-score   support

     Alegria       0.73      0.76      0.75       112
       Miedo       0.77      0.87      0.81       167
    Sorpresa       0.67      0.07      0.13        27
    Tristeza       0.59      0.58      0.59        88

    accuracy                           0.72       394
   macro avg       0.69      0.57      0.57       394
weighted avg       0.71      0.72      0.70       394

```

**AUC ROC (Validación):**
- Macro-average: 0.8633
- Alegria: 0.8897
- Miedo: 0.9071
- Sorpresa: 0.8037
- Tristeza: 0.8468

### Resultados en Prueba
- **Accuracy:** 0.7043
- **Precision (macro):** 0.5634
- **Recall (macro):** 0.5375
- **F1-Score (macro):** 0.5299

```text
              precision    recall  f1-score   support

     Alegria       0.77      0.80      0.78        93
       Miedo       0.77      0.91      0.83       139
    Sorpresa       0.25      0.04      0.07        23
    Tristeza       0.47      0.40      0.43        73

    accuracy                           0.70       328
   macro avg       0.56      0.54      0.53       328
weighted avg       0.66      0.70      0.68       328

```

**AUC ROC (Prueba):**
- Macro-average: 0.8777
- Alegria: 0.9217
- Miedo: 0.9528
- Sorpresa: 0.8128
- Tristeza: 0.8150

### Gráficas de Evaluación Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión CNN](./cm_conjunta_cnn.png)

**Curvas ROC:**

![Curvas ROC CNN](./roc_conjunta_cnn.png)

---

## LSTM

### Resultados de Validación Cruzada (5-Folds en Entrenamiento Efectivo)
- **Accuracy Media:** 0.6565
- **Precision Media (macro):** 0.4831
- **Recall Media (macro):** 0.5146
- **F1-Score Media (macro):** 0.4965

## LSTM

### Resultados en Entrenamiento Efectivo
- **Accuracy:** 0.9171
- **Precision (macro):** 0.9300
- **Recall (macro):** 0.8003
- **F1-Score (macro):** 0.8210

```text
              precision    recall  f1-score   support

     Alegria       0.97      0.95      0.96       262
       Miedo       0.95      0.98      0.96       387
    Sorpresa       1.00      0.33      0.49        64
    Tristeza       0.80      0.95      0.87       204

    accuracy                           0.92       917
   macro avg       0.93      0.80      0.82       917
weighted avg       0.93      0.92      0.91       917

```

**AUC ROC (Entrenamiento Efectivo):**
- Macro-average: 0.9927
- Alegria: 0.9972
- Miedo: 0.9946
- Sorpresa: 0.9872
- Tristeza: 0.9905

### Resultados en Validación
- **Accuracy:** 0.6497
- **Precision (macro):** 0.5598
- **Recall (macro):** 0.5053
- **F1-Score (macro):** 0.5012

```text
              precision    recall  f1-score   support

     Alegria       0.73      0.67      0.70       112
       Miedo       0.68      0.81      0.74       167
    Sorpresa       0.33      0.04      0.07        27
    Tristeza       0.49      0.50      0.50        88

    accuracy                           0.65       394
   macro avg       0.56      0.51      0.50       394
weighted avg       0.63      0.65      0.63       394

```

**AUC ROC (Validación):**
- Macro-average: 0.7943
- Alegria: 0.8378
- Miedo: 0.8413
- Sorpresa: 0.7250
- Tristeza: 0.7662

### Resultados en Prueba
- **Accuracy:** 0.6768
- **Precision (macro):** 0.4957
- **Recall (macro):** 0.5149
- **F1-Score (macro):** 0.5032

```text
              precision    recall  f1-score   support

     Alegria       0.78      0.71      0.74        93
       Miedo       0.72      0.87      0.79       139
    Sorpresa       0.00      0.00      0.00        23
    Tristeza       0.49      0.48      0.48        73

    accuracy                           0.68       328
   macro avg       0.50      0.51      0.50       328
weighted avg       0.63      0.68      0.65       328

```

**AUC ROC (Prueba):**
- Macro-average: 0.8462
- Alegria: 0.9052
- Miedo: 0.9026
- Sorpresa: 0.7792
- Tristeza: 0.7915

### Gráficas de Evaluación Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión LSTM](./cm_conjunta_lstm.png)

**Curvas ROC:**

![Curvas ROC LSTM](./roc_conjunta_lstm.png)

---

