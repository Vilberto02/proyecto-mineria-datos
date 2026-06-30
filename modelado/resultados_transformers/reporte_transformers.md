# Reporte de evaluación de modelos Transformer

Resultados de métricas de rendimiento para los modelos BETO (BERT en Español) y RoBERTa (RoBERTuito).

---

## Comparativa Global (Prueba)

| Métrica           | BETO   | RoBERTa |
| ----------------- | ------ | ------- |
| Accuracy          | 0.7126 | 0.7170  |
| Precision (macro) | 0.7149 | 0.7184  |
| Recall (macro)    | 0.7112 | 0.7165  |
| F1-Score (macro)  | 0.7120 | 0.7167  |
| AUC (macro)       | 0.8963 | 0.9074  |

## BETO

### Resultados en Entrenamiento Efectivo (~56%)

- **Accuracy:** 0.9809
- **Precision (macro):** 0.9812
- **Recall (macro):** 0.9819
- **F1-Score (macro):** 0.9816

```text
              precision    recall  f1-score   support

     Alegria       0.98      0.98      0.98      1304
       Miedo       0.98      0.99      0.98      1122
    Sorpresa       0.99      0.99      0.99      1147
    Tristeza       0.98      0.97      0.97      1503

    accuracy                           0.98      5076
   macro avg       0.98      0.98      0.98      5076
weighted avg       0.98      0.98      0.98      5076
```

**Matriz de Confusión (Entrenamiento):**

![CM BETO Train](./cm_beto_train.png)

**Curva ROC (Entrenamiento — AUC Macro: 0.9982):**

- **AUC Alegria:** 0.9972
- **AUC Miedo:** 0.9989
- **AUC Sorpresa:** 0.9995
- **AUC Tristeza:** 0.9970

![ROC BETO Train](./roc_beto_train.png)

### Resultados en Validación (~24%)

- **Accuracy:** 0.7201
- **Precision (macro):** 0.7193
- **Recall (macro):** 0.7190
- **F1-Score (macro):** 0.7185

```text
              precision    recall  f1-score   support

     Alegria       0.67      0.72      0.70       559
       Miedo       0.76      0.79      0.77       481
    Sorpresa       0.69      0.63      0.66       492
    Tristeza       0.75      0.74      0.75       644

    accuracy                           0.72      2176
   macro avg       0.72      0.72      0.72      2176
weighted avg       0.72      0.72      0.72      2176
```

**Matriz de Confusión (Validación):**

![CM BETO Val](./cm_beto_val.png)

**Curva ROC (Validación — AUC Macro: 0.8997):**

- **AUC Alegria:** 0.8744
- **AUC Miedo:** 0.9309
- **AUC Sorpresa:** 0.8885
- **AUC Tristeza:** 0.9037

![ROC BETO Val](./roc_beto_val.png)

### Resultados en Prueba y Evaluación (20%)

- **Accuracy:** 0.7126
- **Precision (macro):** 0.7149
- **Recall (macro):** 0.7112
- **F1-Score (macro):** 0.7120

```text
              precision    recall  f1-score   support

     Alegria       0.65      0.73      0.69       466
       Miedo       0.79      0.78      0.79       401
    Sorpresa       0.68      0.61      0.64       409
    Tristeza       0.74      0.73      0.73       537

    accuracy                           0.71      1813
   macro avg       0.71      0.71      0.71      1813
weighted avg       0.71      0.71      0.71      1813
```

**Matriz de Confusión (Prueba):**

![CM BETO Test](./cm_beto_test.png)

**Curva ROC (Prueba — AUC Macro: 0.8963):**

- **AUC Alegria:** 0.8661
- **AUC Miedo:** 0.9321
- **AUC Sorpresa:** 0.8868
- **AUC Tristeza:** 0.8988

![ROC BETO Test](./roc_beto_test.png)

## RoBERTa

### Resultados en Entrenamiento Efectivo (~56%)

- **Accuracy:** 0.8229
- **Precision (macro):** 0.8229
- **Recall (macro):** 0.8225
- **F1-Score (macro):** 0.8222

```text
              precision    recall  f1-score   support

     Alegria       0.82      0.76      0.79      1304
       Miedo       0.86      0.86      0.86      1122
    Sorpresa       0.78      0.81      0.79      1147
    Tristeza       0.83      0.86      0.85      1503

    accuracy                           0.82      5076
   macro avg       0.82      0.82      0.82      5076
weighted avg       0.82      0.82      0.82      5076
```

**Matriz de Confusión (Entrenamiento):**

![CM RoBERTa Train](./cm_roberta_train.png)

**Curva ROC (Entrenamiento — AUC Macro: 0.9533):**

- **AUC Alegria:** 0.9340
- **AUC Miedo:** 0.9729
- **AUC Sorpresa:** 0.9545
- **AUC Tristeza:** 0.9513

![ROC RoBERTa Train](./roc_roberta_train.png)

### Resultados en Validación (~24%)

- **Accuracy:** 0.7188
- **Precision (macro):** 0.7165
- **Recall (macro):** 0.7161
- **F1-Score (macro):** 0.7160

```text
              precision    recall  f1-score   support

     Alegria       0.73      0.68      0.70       559
       Miedo       0.76      0.78      0.77       481
    Sorpresa       0.64      0.63      0.63       492
    Tristeza       0.74      0.77      0.76       644

    accuracy                           0.72      2176
   macro avg       0.72      0.72      0.72      2176
weighted avg       0.72      0.72      0.72      2176
```

**Matriz de Confusión (Validación):**

![CM RoBERTa Val](./cm_roberta_val.png)

**Curva ROC (Validación — AUC Macro: 0.8995):**

- **AUC Alegria:** 0.8769
- **AUC Miedo:** 0.9365
- **AUC Sorpresa:** 0.8835
- **AUC Tristeza:** 0.8998

![ROC RoBERTa Val](./roc_roberta_val.png)

### Resultados en Prueba y Evaluación (20%)

- **Accuracy:** 0.7170
- **Precision (macro):** 0.7184
- **Recall (macro):** 0.7165
- **F1-Score (macro):** 0.7167

```text
              precision    recall  f1-score   support

     Alegria       0.73      0.65      0.69       466
       Miedo       0.79      0.80      0.80       401
    Sorpresa       0.65      0.66      0.65       409
    Tristeza       0.71      0.76      0.73       537

    accuracy                           0.72      1813
   macro avg       0.72      0.72      0.72      1813
weighted avg       0.72      0.72      0.72      1813
```

**Matriz de Confusión (Prueba):**

![CM RoBERTa Test](./cm_roberta_test.png)

**Curva ROC (Prueba — AUC Macro: 0.9074):**

- **AUC Alegria:** 0.8760
- **AUC Miedo:** 0.9447
- **AUC Sorpresa:** 0.8990
- **AUC Tristeza:** 0.9087

![ROC RoBERTa Test](./roc_roberta_test.png)
