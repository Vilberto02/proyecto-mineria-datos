# Reporte de Evaluación de Modelos Clásicos

Evaluación robusta utilizando `imblearn.pipeline`, GridSearchCV y SMOTE para evitar Data Leakage y mitigar sobreajuste.

## SVM

**Mejores Hiperparámetros Encontrados:** `{'clf__C': 1}`

### Resultados en el Conjunto de Entrenamiento (70%)

- **Accuracy:** 0.8690
- **Precision (macro):** 0.8697
- **Recall (macro):** 0.8699
- **F1-Score (macro):** 0.8689

```text
              precision    recall  f1-score   support

     Alegria       0.88      0.84      0.86      1630
       Miedo       0.90      0.87      0.89      1403
    Sorpresa       0.80      0.90      0.85      1433
    Tristeza       0.89      0.87      0.88      1879

    accuracy                           0.87      6345
   macro avg       0.87      0.87      0.87      6345
weighted avg       0.87      0.87      0.87      6345

```

### Resultados en el Conjunto de Prueba (30%)

- **Accuracy:** 0.6507
- **Precision (macro):** 0.6583
- **Recall (macro):** 0.6531
- **F1-Score (macro):** 0.6541

```text
              precision    recall  f1-score   support

     Alegria       0.63      0.58      0.60       699
       Miedo       0.79      0.73      0.76       601
    Sorpresa       0.56      0.65      0.60       615
    Tristeza       0.66      0.66      0.66       805

    accuracy                           0.65      2720
   macro avg       0.66      0.65      0.65      2720
weighted avg       0.66      0.65      0.65      2720

```

**Matriz de Confusión:**

![Matriz de Confusión SVM](./cm_svm.png)

## Random Forest

**Mejores Hiperparámetros Encontrados:** `{'clf__max_depth': 30, 'clf__min_samples_split': 5}`

### Resultados en el Conjunto de Entrenamiento (70%)

- **Accuracy:** 0.8594
- **Precision (macro):** 0.8713
- **Recall (macro):** 0.8618
- **F1-Score (macro):** 0.8605

```text
              precision    recall  f1-score   support

     Alegria       0.90      0.80      0.85      1630
       Miedo       0.95      0.85      0.90      1403
    Sorpresa       0.70      0.94      0.80      1433
    Tristeza       0.94      0.85      0.89      1879

    accuracy                           0.86      6345
   macro avg       0.87      0.86      0.86      6345
weighted avg       0.88      0.86      0.86      6345

```

### Resultados en el Conjunto de Prueba (30%)

- **Accuracy:** 0.6312
- **Precision (macro):** 0.6448
- **Recall (macro):** 0.6367
- **F1-Score (macro):** 0.6353

```text
              precision    recall  f1-score   support

     Alegria       0.65      0.51      0.57       699
       Miedo       0.79      0.75      0.77       601
    Sorpresa       0.50      0.67      0.57       615
    Tristeza       0.64      0.62      0.63       805

    accuracy                           0.63      2720
   macro avg       0.64      0.64      0.64      2720
weighted avg       0.64      0.63      0.63      2720

```

**Matriz de Confusión:**

![Matriz de Confusión Random Forest](./cm_random_forest.png)

## Naive Bayes

### Resultados en el Conjunto de Entrenamiento (70%)

- **Accuracy:** 0.7630
- **Precision (macro):** 0.7654
- **Recall (macro):** 0.7639
- **F1-Score (macro):** 0.7624

```text
              precision    recall  f1-score   support

     Alegria       0.78      0.68      0.72      1630
       Miedo       0.81      0.79      0.80      1403
    Sorpresa       0.67      0.80      0.73      1433
    Tristeza       0.80      0.79      0.79      1879

    accuracy                           0.76      6345
   macro avg       0.77      0.76      0.76      6345
weighted avg       0.77      0.76      0.76      6345

```

### Resultados en el Conjunto de Prueba (30%)

- **Accuracy:** 0.6202
- **Precision (macro):** 0.6224
- **Recall (macro):** 0.6205
- **F1-Score (macro):** 0.6196

```text
              precision    recall  f1-score   support

     Alegria       0.64      0.54      0.58       699
       Miedo       0.66      0.66      0.66       601
    Sorpresa       0.56      0.63      0.59       615
    Tristeza       0.63      0.66      0.64       805

    accuracy                           0.62      2720
   macro avg       0.62      0.62      0.62      2720
weighted avg       0.62      0.62      0.62      2720

```

**Matriz de Confusión:**

![Matriz de Confusión Naive Bayes](./cm_naive_bayes.png)
