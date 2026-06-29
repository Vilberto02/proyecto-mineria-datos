# Reporte de Evaluación de Modelos Clásicos

Evaluación robusta utilizando `imblearn.pipeline`, GridSearchCV y SMOTE para evitar Data Leakage y mitigar sobreajuste.

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
- **Accuracy:** 0.8227
- **Precision (macro):** 0.8218
- **Recall (macro):** 0.8255
- **F1-Score (macro):** 0.8230

```text
              precision    recall  f1-score   support

     Alegria       0.81      0.80      0.81      1304
       Miedo       0.84      0.87      0.86      1122
    Sorpresa       0.77      0.83      0.80      1147
    Tristeza       0.86      0.80      0.83      1503

    accuracy                           0.82      5076
   macro avg       0.82      0.83      0.82      5076
weighted avg       0.82      0.82      0.82      5076

```

### Resultados en el Conjunto de Validación (~24%)
- **Accuracy:** 0.6480
- **Precision (macro):** 0.6521
- **Recall (macro):** 0.6502
- **F1-Score (macro):** 0.6498

```text
              precision    recall  f1-score   support

     Alegria       0.62      0.58      0.60       559
       Miedo       0.76      0.74      0.75       481
    Sorpresa       0.53      0.63      0.58       492
    Tristeza       0.70      0.65      0.68       644

    accuracy                           0.65      2176
   macro avg       0.65      0.65      0.65      2176
weighted avg       0.65      0.65      0.65      2176

```

### Resultados en el Conjunto de Prueba y Evaluación (20%)
- **Accuracy:** 0.6514
- **Precision (macro):** 0.6573
- **Recall (macro):** 0.6534
- **F1-Score (macro):** 0.6545

```text
              precision    recall  f1-score   support

     Alegria       0.63      0.62      0.62       466
       Miedo       0.78      0.73      0.75       401
    Sorpresa       0.55      0.62      0.59       409
    Tristeza       0.67      0.64      0.66       537

    accuracy                           0.65      1813
   macro avg       0.66      0.65      0.65      1813
weighted avg       0.66      0.65      0.65      1813

```

**Matriz de Confusión:**

![Matriz de Confusión SVM](./cm_svm.png)

### Curva ROC y AUC (One-vs-Rest)

- **AUC Macro-average:** 0.8451
- **AUC Alegria:** 0.8188
- **AUC Miedo:** 0.8960
- **AUC Sorpresa:** 0.8339
- **AUC Tristeza:** 0.8303

![Curva ROC SVM](./roc_svm.png)

---

## Random Forest

**Mejores Hiperparámetros Encontrados:** `{'clf__max_depth': 30, 'clf__min_samples_split': 10}`

### Resultados en el Conjunto de Entrenamiento Efectivo (~56%)
- **Accuracy:** 0.8030
- **Precision (macro):** 0.8215
- **Recall (macro):** 0.8069
- **F1-Score (macro):** 0.8059

```text
              precision    recall  f1-score   support

     Alegria       0.84      0.74      0.79      1304
       Miedo       0.91      0.81      0.86      1122
    Sorpresa       0.63      0.89      0.74      1147
    Tristeza       0.91      0.78      0.84      1503

    accuracy                           0.80      5076
   macro avg       0.82      0.81      0.81      5076
weighted avg       0.83      0.80      0.81      5076

```

### Resultados en el Conjunto de Validación (~24%)
- **Accuracy:** 0.6392
- **Precision (macro):** 0.6504
- **Recall (macro):** 0.6444
- **F1-Score (macro):** 0.6429

```text
              precision    recall  f1-score   support

     Alegria       0.63      0.53      0.58       559
       Miedo       0.78      0.75      0.76       481
    Sorpresa       0.50      0.67      0.57       492
    Tristeza       0.69      0.63      0.66       644

    accuracy                           0.64      2176
   macro avg       0.65      0.64      0.64      2176
weighted avg       0.65      0.64      0.64      2176

```

### Resultados en el Conjunto de Prueba y Evaluación (20%)
- **Accuracy:** 0.6453
- **Precision (macro):** 0.6593
- **Recall (macro):** 0.6513
- **F1-Score (macro):** 0.6510

```text
              precision    recall  f1-score   support

     Alegria       0.65      0.57      0.61       466
       Miedo       0.81      0.74      0.77       401
    Sorpresa       0.52      0.69      0.59       409
    Tristeza       0.66      0.61      0.63       537

    accuracy                           0.65      1813
   macro avg       0.66      0.65      0.65      1813
weighted avg       0.66      0.65      0.65      1813

```

**Matriz de Confusión:**

![Matriz de Confusión Random Forest](./cm_random_forest.png)

### Curva ROC y AUC (One-vs-Rest)

- **AUC Macro-average:** 0.8496
- **AUC Alegria:** 0.8240
- **AUC Miedo:** 0.9023
- **AUC Sorpresa:** 0.8330
- **AUC Tristeza:** 0.8379

![Curva ROC Random Forest](./roc_random_forest.png)

---

## Naive Bayes

### Resultados en el Conjunto de Entrenamiento Efectivo (~56%)
- **Accuracy:** 0.7486
- **Precision (macro):** 0.7498
- **Recall (macro):** 0.7494
- **F1-Score (macro):** 0.7485

```text
              precision    recall  f1-score   support

     Alegria       0.74      0.71      0.73      1304
       Miedo       0.80      0.77      0.78      1122
    Sorpresa       0.66      0.76      0.71      1147
    Tristeza       0.79      0.76      0.77      1503

    accuracy                           0.75      5076
   macro avg       0.75      0.75      0.75      5076
weighted avg       0.75      0.75      0.75      5076

```

### Resultados en el Conjunto de Validación (~24%)
- **Accuracy:** 0.6176
- **Precision (macro):** 0.6199
- **Recall (macro):** 0.6170
- **F1-Score (macro):** 0.6170

```text
              precision    recall  f1-score   support

     Alegria       0.63      0.59      0.61       559
       Miedo       0.66      0.61      0.63       481
    Sorpresa       0.53      0.63      0.58       492
    Tristeza       0.65      0.64      0.65       644

    accuracy                           0.62      2176
   macro avg       0.62      0.62      0.62      2176
weighted avg       0.62      0.62      0.62      2176

```

### Resultados en el Conjunto de Prueba y Evaluación (20%)
- **Accuracy:** 0.6304
- **Precision (macro):** 0.6320
- **Recall (macro):** 0.6300
- **F1-Score (macro):** 0.6304

```text
              precision    recall  f1-score   support

     Alegria       0.63      0.62      0.63       466
       Miedo       0.68      0.63      0.65       401
    Sorpresa       0.57      0.63      0.60       409
    Tristeza       0.64      0.64      0.64       537

    accuracy                           0.63      1813
   macro avg       0.63      0.63      0.63      1813
weighted avg       0.63      0.63      0.63      1813

```

**Matriz de Confusión:**

![Matriz de Confusión Naive Bayes](./cm_naive_bayes.png)

### Curva ROC y AUC (One-vs-Rest)

- **AUC Macro-average:** 0.8402
- **AUC Alegria:** 0.8205
- **AUC Miedo:** 0.8587
- **AUC Sorpresa:** 0.8496
- **AUC Tristeza:** 0.8306

![Curva ROC Naive Bayes](./roc_naive_bayes.png)

---

