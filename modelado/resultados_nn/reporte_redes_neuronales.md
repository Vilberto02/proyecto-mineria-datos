# Reporte de Evaluación de Redes Neuronales

Resultados de métricas de rendimiento para los modelos CNN y LSTM.

## CNN

### Resultados de Validación Cruzada (5-Folds en Entrenamiento Efectivo)
- **Accuracy Media:** 0.5544
- **Precision Media (macro):** 0.5600
- **Recall Media (macro):** 0.5528
- **F1-Score Media (macro):** 0.5544

## CNN

### Resultados en Entrenamiento Efectivo
- **Accuracy:** 0.9868
- **Precision (macro):** 0.9868
- **Recall (macro):** 0.9872
- **F1-Score (macro):** 0.9870

```text
              precision    recall  f1-score   support

     Alegria       0.98      0.99      0.98      1304
       Miedo       0.99      0.99      0.99      1122
    Sorpresa       0.99      0.99      0.99      1147
    Tristeza       0.99      0.98      0.99      1503

    accuracy                           0.99      5076
   macro avg       0.99      0.99      0.99      5076
weighted avg       0.99      0.99      0.99      5076

```

**AUC ROC (Entrenamiento Efectivo):**
- Macro-average: 0.9991
- Alegria: 0.9988
- Miedo: 0.9997
- Sorpresa: 0.9994
- Tristeza: 0.9985

### Resultados en Validación
- **Accuracy:** 0.5653
- **Precision (macro):** 0.5713
- **Recall (macro):** 0.5653
- **F1-Score (macro):** 0.5678

```text
              precision    recall  f1-score   support

     Alegria       0.50      0.54      0.52       559
       Miedo       0.74      0.68      0.71       481
    Sorpresa       0.45      0.45      0.45       492
    Tristeza       0.60      0.58      0.59       644

    accuracy                           0.57      2176
   macro avg       0.57      0.57      0.57      2176
weighted avg       0.57      0.57      0.57      2176

```

**AUC ROC (Validación):**
- Macro-average: 0.7986
- Alegria: 0.7554
- Miedo: 0.8709
- Sorpresa: 0.7664
- Tristeza: 0.8006

### Resultados en Prueba
- **Accuracy:** 0.5742
- **Precision (macro):** 0.5845
- **Recall (macro):** 0.5728
- **F1-Score (macro):** 0.5770

```text
              precision    recall  f1-score   support

     Alegria       0.50      0.58      0.54       466
       Miedo       0.76      0.65      0.70       401
    Sorpresa       0.48      0.47      0.48       409
    Tristeza       0.59      0.60      0.59       537

    accuracy                           0.57      1813
   macro avg       0.58      0.57      0.58      1813
weighted avg       0.58      0.57      0.58      1813

```

**AUC ROC (Prueba):**
- Macro-average: 0.8014
- Alegria: 0.7593
- Miedo: 0.8713
- Sorpresa: 0.7833
- Tristeza: 0.7903

### Gráficas de Evaluación Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión CNN](./cm_conjunta_cnn.png)

**Curvas ROC:**

![Curvas ROC CNN](./roc_conjunta_cnn.png)

---

## LSTM

### Resultados de Validación Cruzada (5-Folds en Entrenamiento Efectivo)
- **Accuracy Media:** 0.5447
- **Precision Media (macro):** 0.5552
- **Recall Media (macro):** 0.5441
- **F1-Score Media (macro):** 0.5463

## LSTM

### Resultados en Entrenamiento Efectivo
- **Accuracy:** 0.8304
- **Precision (macro):** 0.8306
- **Recall (macro):** 0.8312
- **F1-Score (macro):** 0.8289

```text
              precision    recall  f1-score   support

     Alegria       0.89      0.77      0.83      1304
       Miedo       0.79      0.88      0.83      1122
    Sorpresa       0.79      0.83      0.81      1147
    Tristeza       0.85      0.85      0.85      1503

    accuracy                           0.83      5076
   macro avg       0.83      0.83      0.83      5076
weighted avg       0.83      0.83      0.83      5076

```

**AUC ROC (Entrenamiento Efectivo):**
- Macro-average: 0.9579
- Alegria: 0.9567
- Miedo: 0.9649
- Sorpresa: 0.9482
- Tristeza: 0.9617

### Resultados en Validación
- **Accuracy:** 0.5625
- **Precision (macro):** 0.5572
- **Recall (macro):** 0.5649
- **F1-Score (macro):** 0.5585

```text
              precision    recall  f1-score   support

     Alegria       0.56      0.48      0.52       559
       Miedo       0.61      0.75      0.67       481
    Sorpresa       0.48      0.45      0.46       492
    Tristeza       0.59      0.58      0.58       644

    accuracy                           0.56      2176
   macro avg       0.56      0.56      0.56      2176
weighted avg       0.56      0.56      0.56      2176

```

**AUC ROC (Validación):**
- Macro-average: 0.7911
- Alegria: 0.7686
- Miedo: 0.8573
- Sorpresa: 0.7609
- Tristeza: 0.7765

### Resultados en Prueba
- **Accuracy:** 0.5422
- **Precision (macro):** 0.5385
- **Recall (macro):** 0.5457
- **F1-Score (macro):** 0.5394

```text
              precision    recall  f1-score   support

     Alegria       0.55      0.46      0.50       466
       Miedo       0.61      0.74      0.67       401
    Sorpresa       0.46      0.44      0.45       409
    Tristeza       0.53      0.55      0.54       537

    accuracy                           0.54      1813
   macro avg       0.54      0.55      0.54      1813
weighted avg       0.54      0.54      0.54      1813

```

**AUC ROC (Prueba):**
- Macro-average: 0.7792
- Alegria: 0.7654
- Miedo: 0.8572
- Sorpresa: 0.7347
- Tristeza: 0.7581

### Gráficas de Evaluación Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión LSTM](./cm_conjunta_lstm.png)

**Curvas ROC:**

![Curvas ROC LSTM](./roc_conjunta_lstm.png)

---

