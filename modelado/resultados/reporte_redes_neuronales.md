# Reporte de Evaluación de Redes Neuronales Profundas

Resultados de las métricas de rendimiento para los modelos CNN y LSTM.

## CNN

- **Accuracy:** 0.5078
- **Precision (macro):** 0.3951
- **Recall (macro):** 0.3967
- **F1-Score (macro):** 0.3794

```text
              precision    recall  f1-score   support

     Alegría       0.54      0.48      0.51       699
        Asco       0.17      0.03      0.05       241
         Ira       0.11      0.02      0.03       170
       Miedo       0.66      0.75      0.70       601
    Sorpresa       0.39      0.50      0.44       615
    Tristeza       0.51      0.60      0.55       805

    accuracy                           0.51      3131
   macro avg       0.40      0.40      0.38      3131
weighted avg       0.47      0.51      0.48      3131

```

**Matriz de Confusión:**

![Matriz de Confusión CNN](file:///C:/Users/Usuario/Desktop/Mineria-Datos/Proyecto-Minería-G4/modelado/resultados/cm_nn_cnn.png)

## LSTM

- **Accuracy:** 0.4941
- **Precision (macro):** 0.3902
- **Recall (macro):** 0.3874
- **F1-Score (macro):** 0.3767

```text
              precision    recall  f1-score   support

     Alegría       0.51      0.46      0.48       699
        Asco       0.17      0.04      0.06       241
         Ira       0.07      0.02      0.04       170
       Miedo       0.71      0.68      0.70       601
    Sorpresa       0.36      0.53      0.43       615
    Tristeza       0.52      0.60      0.56       805

    accuracy                           0.49      3131
   macro avg       0.39      0.39      0.38      3131
weighted avg       0.47      0.49      0.48      3131

```

**Matriz de Confusión:**

![Matriz de Confusión LSTM](file:///C:/Users/Usuario/Desktop/Mineria-Datos/Proyecto-Minería-G4/modelado/resultados/cm_nn_lstm.png)

