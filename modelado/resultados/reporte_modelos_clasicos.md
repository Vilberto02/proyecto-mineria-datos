# Reporte de Evaluación de Modelos Clásicos

Resultados de las métricas de rendimiento de los diferentes algoritmos clásicos (SVM, Naive Bayes, Árbol de Decisión y Random Forest).

## SVM

### Resultados en el Conjunto de Entrenamiento (70%)
- **Accuracy:** 0.7511
- **Precision (macro):** 0.7945
- **Recall (macro):** 0.6345
- **F1-Score (macro):** 0.6543

```text
              precision    recall  f1-score   support

     Alegría       0.76      0.80      0.78      1630
        Asco       0.83      0.27      0.41       562
         Ira       0.92      0.24      0.38       398
       Miedo       0.85      0.81      0.83      1403
    Sorpresa       0.65      0.84      0.73      1433
    Tristeza       0.75      0.85      0.80      1879

    accuracy                           0.75      7305
   macro avg       0.79      0.63      0.65      7305
weighted avg       0.77      0.75      0.73      7305

```

### Resultados en el Conjunto de Prueba (30%)
- **Accuracy:** 0.5561
- **Precision (macro):** 0.4328
- **Recall (macro):** 0.4336
- **F1-Score (macro):** 0.4178

```text
              precision    recall  f1-score   support

     Alegría       0.55      0.57      0.56       699
        Asco       0.28      0.06      0.10       241
         Ira       0.00      0.00      0.00       170
       Miedo       0.75      0.69      0.72       601
    Sorpresa       0.45      0.62      0.52       615
    Tristeza       0.56      0.65      0.60       805

    accuracy                           0.56      3131
   macro avg       0.43      0.43      0.42      3131
weighted avg       0.52      0.56      0.53      3131

```

**Matriz de Confusión:**

![Matriz de Confusión SVM](file:///C:/Users/Usuario/Desktop/Mineria-Datos/Proyecto-Minería-G4/modelado/resultados/cm_svm.png)

## Naive Bayes

### Resultados en el Conjunto de Entrenamiento (70%)
- **Accuracy:** 0.6241
- **Precision (macro):** 0.7775
- **Recall (macro):** 0.4703
- **F1-Score (macro):** 0.4476

```text
              precision    recall  f1-score   support

     Alegría       0.68      0.70      0.69      1630
        Asco       1.00      0.00      0.01       562
         Ira       1.00      0.00      0.01       398
       Miedo       0.82      0.60      0.69      1403
    Sorpresa       0.64      0.62      0.63      1433
    Tristeza       0.52      0.90      0.66      1879

    accuracy                           0.62      7305
   macro avg       0.78      0.47      0.45      7305
weighted avg       0.70      0.62      0.58      7305

```

### Resultados en el Conjunto de Prueba (30%)
- **Accuracy:** 0.4966
- **Precision (macro):** 0.3628
- **Recall (macro):** 0.3682
- **F1-Score (macro):** 0.3489

```text
              precision    recall  f1-score   support

     Alegría       0.56      0.55      0.55       699
        Asco       0.00      0.00      0.00       241
         Ira       0.00      0.00      0.00       170
       Miedo       0.72      0.46      0.56       601
    Sorpresa       0.48      0.38      0.42       615
    Tristeza       0.42      0.82      0.56       805

    accuracy                           0.50      3131
   macro avg       0.36      0.37      0.35      3131
weighted avg       0.46      0.50      0.46      3131

```

**Matriz de Confusión:**

![Matriz de Confusión Naive Bayes](file:///C:/Users/Usuario/Desktop/Mineria-Datos/Proyecto-Minería-G4/modelado/resultados/cm_naive_bayes.png)

## Árbol de Decisión

### Resultados en el Conjunto de Entrenamiento (70%)
- **Accuracy:** 0.9906
- **Precision (macro):** 0.9895
- **Recall (macro):** 0.9930
- **F1-Score (macro):** 0.9912

```text
              precision    recall  f1-score   support

     Alegría       0.98      1.00      0.99      1630
        Asco       0.99      1.00      0.99       562
         Ira       0.98      1.00      0.99       398
       Miedo       0.99      1.00      0.99      1403
    Sorpresa       1.00      1.00      1.00      1433
    Tristeza       1.00      0.97      0.99      1879

    accuracy                           0.99      7305
   macro avg       0.99      0.99      0.99      7305
weighted avg       0.99      0.99      0.99      7305

```

### Resultados en el Conjunto de Prueba (30%)
- **Accuracy:** 0.4538
- **Precision (macro):** 0.3726
- **Recall (macro):** 0.3738
- **F1-Score (macro):** 0.3727

```text
              precision    recall  f1-score   support

     Alegría       0.47      0.48      0.47       699
        Asco       0.14      0.12      0.13       241
         Ira       0.12      0.09      0.10       170
       Miedo       0.63      0.65      0.64       601
    Sorpresa       0.37      0.40      0.38       615
    Tristeza       0.51      0.50      0.51       805

    accuracy                           0.45      3131
   macro avg       0.37      0.37      0.37      3131
weighted avg       0.45      0.45      0.45      3131

```

**Matriz de Confusión:**

![Matriz de Confusión Árbol de Decisión](file:///C:/Users/Usuario/Desktop/Mineria-Datos/Proyecto-Minería-G4/modelado/resultados/cm_arbol_de_decisión.png)

## Random Forest

### Resultados en el Conjunto de Entrenamiento (70%)
- **Accuracy:** 0.9906
- **Precision (macro):** 0.9927
- **Recall (macro):** 0.9896
- **F1-Score (macro):** 0.9911

```text
              precision    recall  f1-score   support

     Alegría       0.99      0.99      0.99      1630
        Asco       1.00      0.99      0.99       562
         Ira       1.00      0.98      0.99       398
       Miedo       1.00      0.99      0.99      1403
    Sorpresa       1.00      1.00      1.00      1433
    Tristeza       0.98      0.99      0.99      1879

    accuracy                           0.99      7305
   macro avg       0.99      0.99      0.99      7305
weighted avg       0.99      0.99      0.99      7305

```

### Resultados en el Conjunto de Prueba (30%)
- **Accuracy:** 0.5487
- **Precision (macro):** 0.4143
- **Recall (macro):** 0.4219
- **F1-Score (macro):** 0.3986

```text
              precision    recall  f1-score   support

     Alegría       0.55      0.55      0.55       699
        Asco       0.22      0.01      0.02       241
         Ira       0.00      0.00      0.00       170
       Miedo       0.76      0.74      0.75       601
    Sorpresa       0.42      0.56      0.48       615
    Tristeza       0.53      0.68      0.59       805

    accuracy                           0.55      3131
   macro avg       0.41      0.42      0.40      3131
weighted avg       0.51      0.55      0.52      3131

```

**Matriz de Confusión:**

![Matriz de Confusión Random Forest](file:///C:/Users/Usuario/Desktop/Mineria-Datos/Proyecto-Minería-G4/modelado/resultados/cm_random_forest.png)

