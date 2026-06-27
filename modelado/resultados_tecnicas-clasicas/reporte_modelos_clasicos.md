# Reporte de Evaluación de Modelos Clásicos

Evaluación robusta utilizando `imblearn.pipeline`, GridSearchCV y SMOTE para evitar Data Leakage y mitigar sobreajuste.

## SVM

**Mejores Hiperparámetros Encontrados:** `{'clf__estimator__C': 1}`

### Resultados en el Conjunto de Entrenamiento (70%)
- **Accuracy:** 0.8257
- **Precision (macro):** 0.8249
- **Recall (macro):** 0.8279
- **F1-Score (macro):** 0.8258

```text
              precision    recall  f1-score   support

     Alegria       0.83      0.80      0.82      1630
       Miedo       0.84      0.86      0.85      1403
    Sorpresa       0.77      0.84      0.80      1433
    Tristeza       0.86      0.81      0.83      1879

    accuracy                           0.83      6345
   macro avg       0.82      0.83      0.83      6345
weighted avg       0.83      0.83      0.83      6345

```

### Resultados en el Conjunto de Prueba (30%)
- **Accuracy:** 0.6507
- **Precision (macro):** 0.6553
- **Recall (macro):** 0.6534
- **F1-Score (macro):** 0.6538

```text
              precision    recall  f1-score   support

     Alegria       0.62      0.60      0.61       699
       Miedo       0.78      0.74      0.76       601
    Sorpresa       0.56      0.62      0.59       615
    Tristeza       0.66      0.64      0.65       805

    accuracy                           0.65      2720
   macro avg       0.66      0.65      0.65      2720
weighted avg       0.65      0.65      0.65      2720

```

**Matriz de Confusión:**

![Matriz de Confusión SVM](./cm_svm.png)

### Curva ROC y AUC (One-vs-Rest)

- **AUC Macro-average:** 0.8402
- **AUC Alegria:** 0.8163
- **AUC Miedo:** 0.8937
- **AUC Sorpresa:** 0.8233
- **AUC Tristeza:** 0.8266

![Curva ROC SVM](./roc_svm.png)

---

## Random Forest

**Mejores Hiperparámetros Encontrados:** `{'clf__max_depth': 30, 'clf__min_samples_split': 10}`

### Resultados en el Conjunto de Entrenamiento (70%)
- **Accuracy:** 0.7965
- **Precision (macro):** 0.8168
- **Recall (macro):** 0.8002
- **F1-Score (macro):** 0.7998

```text
              precision    recall  f1-score   support

     Alegria       0.84      0.73      0.78      1630
       Miedo       0.92      0.81      0.86      1403
    Sorpresa       0.61      0.88      0.72      1433
    Tristeza       0.90      0.78      0.84      1879

    accuracy                           0.80      6345
   macro avg       0.82      0.80      0.80      6345
weighted avg       0.82      0.80      0.80      6345

```

### Resultados en el Conjunto de Prueba (30%)
- **Accuracy:** 0.6408
- **Precision (macro):** 0.6547
- **Recall (macro):** 0.6467
- **F1-Score (macro):** 0.6456

```text
              precision    recall  f1-score   support

     Alegria       0.64      0.54      0.59       699
       Miedo       0.79      0.75      0.77       601
    Sorpresa       0.50      0.68      0.58       615
    Tristeza       0.68      0.62      0.65       805

    accuracy                           0.64      2720
   macro avg       0.65      0.65      0.65      2720
weighted avg       0.66      0.64      0.64      2720

```

**Matriz de Confusión:**

![Matriz de Confusión Random Forest](./cm_random_forest.png)

### Curva ROC y AUC (One-vs-Rest)

- **AUC Macro-average:** 0.8430
- **AUC Alegria:** 0.8162
- **AUC Miedo:** 0.8987
- **AUC Sorpresa:** 0.8221
- **AUC Tristeza:** 0.8342

![Curva ROC Random Forest](./roc_random_forest.png)

---

## Naive Bayes

### Resultados en el Conjunto de Entrenamiento (70%)
- **Accuracy:** 0.7521
- **Precision (macro):** 0.7537
- **Recall (macro):** 0.7523
- **F1-Score (macro):** 0.7517

```text
              precision    recall  f1-score   support

     Alegria       0.74      0.72      0.73      1630
       Miedo       0.80      0.75      0.78      1403
    Sorpresa       0.67      0.78      0.72      1433
    Tristeza       0.80      0.76      0.78      1879

    accuracy                           0.75      6345
   macro avg       0.75      0.75      0.75      6345
weighted avg       0.76      0.75      0.75      6345

```

### Resultados en el Conjunto de Prueba (30%)
- **Accuracy:** 0.6125
- **Precision (macro):** 0.6147
- **Recall (macro):** 0.6118
- **F1-Score (macro):** 0.6125

```text
              precision    recall  f1-score   support

     Alegria       0.62      0.57      0.59       699
       Miedo       0.66      0.62      0.64       601
    Sorpresa       0.56      0.61      0.59       615
    Tristeza       0.61      0.64      0.63       805

    accuracy                           0.61      2720
   macro avg       0.61      0.61      0.61      2720
weighted avg       0.61      0.61      0.61      2720

```

**Matriz de Confusión:**

![Matriz de Confusión Naive Bayes](./cm_naive_bayes.png)

### Curva ROC y AUC (One-vs-Rest)

- **AUC Macro-average:** 0.8356
- **AUC Alegria:** 0.8142
- **AUC Miedo:** 0.8644
- **AUC Sorpresa:** 0.8356
- **AUC Tristeza:** 0.8273

![Curva ROC Naive Bayes](./roc_naive_bayes.png)

---

