# Reporte de Evaluación de Modelos Clásicos

## División de Datos

| Conjunto | Registros | % del total |
|---|---|---|
| Entrenamiento Efectivo | 5076 | ~56.0% |
| Validación | 2176 | ~24.0% |
| Prueba y Evaluación | 1813 | ~20.0% |
| **Total** | **9065** | **100%** |

## Distribución por Emoción

| Emoción | Entren. Efectivo | Validación | Prueba |
|---|---|---|---|
| Alegria | 1304 | 559 | 466 |
| Miedo | 1122 | 481 | 401 |
| Sorpresa | 1147 | 492 | 409 |
| Tristeza | 1503 | 644 | 537 |
| **Total** | **5076** | **2176** | **1813** |

---

## SVM

**Mejores Hiperparámetros Encontrados:** `{'clf__estimator__C': 1}`

### Resultados en el Conjunto de Entrenamiento Efectivo (~56%)
- **Accuracy:** 0.8008
- **Precision (macro):** 0.8016
- **Recall (macro):** 0.8028
- **F1-Score (macro):** 0.8016

```text
              precision    recall  f1-score   support

     Alegria       0.79      0.77      0.78      1304
       Miedo       0.85      0.84      0.85      1122
    Sorpresa       0.73      0.81      0.77      1147
    Tristeza       0.83      0.79      0.81      1503

    accuracy                           0.80      5076
   macro avg       0.80      0.80      0.80      5076
weighted avg       0.80      0.80      0.80      5076

```

**AUC ROC (Entrenamiento):**
- Macro-average: 0.9396
- Alegria: 0.9264
- Miedo: 0.9629
- Sorpresa: 0.9375
- Tristeza: 0.9312

### Resultados en el Conjunto de Validación (~24%)
- **Accuracy:** 0.6475
- **Precision (macro):** 0.6520
- **Recall (macro):** 0.6482
- **F1-Score (macro):** 0.6493

```text
              precision    recall  f1-score   support

     Alegria       0.61      0.60      0.61       559
       Miedo       0.77      0.72      0.75       481
    Sorpresa       0.54      0.60      0.57       492
    Tristeza       0.69      0.66      0.68       644

    accuracy                           0.65      2176
   macro avg       0.65      0.65      0.65      2176
weighted avg       0.65      0.65      0.65      2176

```

**AUC ROC (Validación):**
- Macro-average: 0.8428
- Alegria: 0.8072
- Miedo: 0.8910
- Sorpresa: 0.8262
- Tristeza: 0.8456

### Resultados en el Conjunto de Prueba y Evaluación (20%)
- **Accuracy:** 0.6442
- **Precision (macro):** 0.6514
- **Recall (macro):** 0.6446
- **F1-Score (macro):** 0.6474

```text
              precision    recall  f1-score   support

     Alegria       0.60      0.61      0.61       466
       Miedo       0.80      0.72      0.76       401
    Sorpresa       0.56      0.59      0.57       409
    Tristeza       0.65      0.66      0.65       537

    accuracy                           0.64      1813
   macro avg       0.65      0.64      0.65      1813
weighted avg       0.65      0.64      0.65      1813

```

**AUC ROC (Prueba):**
- Macro-average: 0.8451
- Alegria: 0.8089
- Miedo: 0.9007
- Sorpresa: 0.8372
- Tristeza: 0.8324

### Gráficas de Evaluación Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión SVM](./cm_conjunta_svm.png)

**Curvas ROC:**

![Curvas ROC SVM](./roc_conjunta_svm.png)

---

## Random Forest

**Mejores Hiperparámetros Encontrados:** `{'clf__max_depth': 30, 'clf__min_samples_split': 5}`

### Resultados en el Conjunto de Entrenamiento Efectivo (~56%)
- **Accuracy:** 0.8172
- **Precision (macro):** 0.8387
- **Recall (macro):** 0.8232
- **F1-Score (macro):** 0.8207

```text
              precision    recall  f1-score   support

     Alegria       0.87      0.76      0.81      1304
       Miedo       0.92      0.84      0.88      1122
    Sorpresa       0.64      0.93      0.76      1147
    Tristeza       0.92      0.77      0.84      1503

    accuracy                           0.82      5076
   macro avg       0.84      0.82      0.82      5076
weighted avg       0.85      0.82      0.82      5076

```

**AUC ROC (Entrenamiento):**
- Macro-average: 0.9718
- Alegria: 0.9678
- Miedo: 0.9841
- Sorpresa: 0.9641
- Tristeza: 0.9710

### Resultados en el Conjunto de Validación (~24%)
- **Accuracy:** 0.6576
- **Precision (macro):** 0.6719
- **Recall (macro):** 0.6615
- **F1-Score (macro):** 0.6607

```text
              precision    recall  f1-score   support

     Alegria       0.67      0.56      0.61       559
       Miedo       0.78      0.73      0.75       481
    Sorpresa       0.51      0.70      0.59       492
    Tristeza       0.73      0.66      0.69       644

    accuracy                           0.66      2176
   macro avg       0.67      0.66      0.66      2176
weighted avg       0.68      0.66      0.66      2176

```

**AUC ROC (Validación):**
- Macro-average: 0.8535
- Alegria: 0.8138
- Miedo: 0.9012
- Sorpresa: 0.8359
- Tristeza: 0.8618

### Resultados en el Conjunto de Prueba y Evaluación (20%)
- **Accuracy:** 0.6376
- **Precision (macro):** 0.6575
- **Recall (macro):** 0.6444
- **F1-Score (macro):** 0.6440

```text
              precision    recall  f1-score   support

     Alegria       0.65      0.54      0.59       466
       Miedo       0.82      0.73      0.77       401
    Sorpresa       0.50      0.71      0.58       409
    Tristeza       0.66      0.60      0.63       537

    accuracy                           0.64      1813
   macro avg       0.66      0.64      0.64      1813
weighted avg       0.66      0.64      0.64      1813

```

**AUC ROC (Prueba):**
- Macro-average: 0.8502
- Alegria: 0.8216
- Miedo: 0.9065
- Sorpresa: 0.8336
- Tristeza: 0.8381

### Gráficas de Evaluación Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión Random Forest](./cm_conjunta_random_forest.png)

**Curvas ROC:**

![Curvas ROC Random Forest](./roc_conjunta_random_forest.png)

---

## Naive Bayes

### Resultados en el Conjunto de Entrenamiento Efectivo (~56%)
- **Accuracy:** 0.7118
- **Precision (macro):** 0.7245
- **Recall (macro):** 0.7043
- **F1-Score (macro):** 0.7099

```text
              precision    recall  f1-score   support

     Alegria       0.70      0.69      0.69      1304
       Miedo       0.85      0.66      0.74      1122
    Sorpresa       0.67      0.65      0.66      1147
    Tristeza       0.68      0.82      0.74      1503

    accuracy                           0.71      5076
   macro avg       0.72      0.70      0.71      5076
weighted avg       0.72      0.71      0.71      5076

```

**AUC ROC (Entrenamiento):**
- Macro-average: 0.9028
- Alegria: 0.8877
- Miedo: 0.9250
- Sorpresa: 0.8962
- Tristeza: 0.9021

### Resultados en el Conjunto de Validación (~24%)
- **Accuracy:** 0.6048
- **Precision (macro):** 0.6187
- **Recall (macro):** 0.5937
- **F1-Score (macro):** 0.5986

```text
              precision    recall  f1-score   support

     Alegria       0.61      0.60      0.61       559
       Miedo       0.73      0.51      0.60       481
    Sorpresa       0.55      0.52      0.54       492
    Tristeza       0.58      0.74      0.65       644

    accuracy                           0.60      2176
   macro avg       0.62      0.59      0.60      2176
weighted avg       0.62      0.60      0.60      2176

```

**AUC ROC (Validación):**
- Macro-average: 0.8349
- Alegria: 0.8164
- Miedo: 0.8512
- Sorpresa: 0.8287
- Tristeza: 0.8422

### Resultados en el Conjunto de Prueba y Evaluación (20%)
- **Accuracy:** 0.6139
- **Precision (macro):** 0.6374
- **Recall (macro):** 0.6047
- **F1-Score (macro):** 0.6121

```text
              precision    recall  f1-score   support

     Alegria       0.59      0.60      0.60       466
       Miedo       0.79      0.54      0.64       401
    Sorpresa       0.60      0.55      0.57       409
    Tristeza       0.57      0.73      0.64       537

    accuracy                           0.61      1813
   macro avg       0.64      0.60      0.61      1813
weighted avg       0.63      0.61      0.61      1813

```

**AUC ROC (Prueba):**
- Macro-average: 0.8370
- Alegria: 0.8132
- Miedo: 0.8581
- Sorpresa: 0.8477
- Tristeza: 0.8278

### Gráficas de Evaluación Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión Naive Bayes](./cm_conjunta_naive_bayes.png)

**Curvas ROC:**

![Curvas ROC Naive Bayes](./roc_conjunta_naive_bayes.png)

---

