# Reporte de evaluación de modelos Transformer

## BETO

### Resultados en Entrenamiento Efectivo

- **Accuracy:** 0.9804
- **Precision (macro):** 0.9694
- **Recall (macro):** 0.9851
- **F1-Score (macro):** 0.9768

**AUC ROC (Entrenamiento Efectivo):**

- Macro-promedio: 0.9992
- Alegria: 0.9992
- Miedo: 0.9980
- Sorpresa: 0.9998
- Tristeza: 0.9996

**Reporte por Clase (Entrenamiento Efectivo)**

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.9846    | 0.9733 | 0.9789   | 262     |
| Miedo    | 0.9895    | 0.9767 | 0.9831   | 387     |
| Sorpresa | 0.9275    | 1.0000 | 0.9624   | 64      |
| Tristeza | 0.9758    | 0.9902 | 0.9830   | 204     |

### Resultados en Validación

- **Accuracy:** 0.7995
- **Precision (macro):** 0.7526
- **Recall (macro):** 0.7319
- **F1-Score (macro):** 0.7405

**AUC ROC (Validación):**

- Macro-promedio: 0.9233
- Alegria: 0.9430
- Miedo: 0.9534
- Sorpresa: 0.8784
- Tristeza: 0.9122

**Reporte por Clase (Validación)**

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.8083    | 0.8661 | 0.8362   | 112     |
| Miedo    | 0.8663    | 0.8922 | 0.8791   | 167     |
| Sorpresa | 0.6522    | 0.5556 | 0.6000   | 27      |
| Tristeza | 0.6835    | 0.6136 | 0.6467   | 88      |

### Resultados en Prueba

- **Accuracy:** 0.8049
- **Precision (macro):** 0.7427
- **Recall (macro):** 0.7471
- **F1-Score (macro):** 0.7435

**AUC ROC (Prueba):**

- Macro-promedio: 0.9365
- Alegria: 0.9472
- Miedo: 0.9747
- Sorpresa: 0.9254
- Tristeza: 0.8912

**Reporte por Clase (Prueba)**

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.8421    | 0.8602 | 0.8511   | 93      |
| Miedo    | 0.8836    | 0.9281 | 0.9053   | 139     |
| Sorpresa | 0.6000    | 0.6522 | 0.6250   | 23      |
| Tristeza | 0.6452    | 0.5479 | 0.5926   | 73      |

### Gráficas Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión BETO](./cm_conjunta_beto.png)

**Curvas ROC:**

![Curvas ROC BETO](./roc_conjunta_beto.png)

---

## RoBERTuito

### Resultados en Entrenamiento Efectivo

- **Accuracy:** 0.8713
- **Precision (macro):** 0.8299
- **Recall (macro):** 0.8792
- **F1-Score (macro):** 0.8473

**AUC ROC (Entrenamiento Efectivo):**

- Macro-promedio: 0.9751
- Alegria: 0.9774
- Miedo: 0.9772
- Sorpresa: 0.9843
- Tristeza: 0.9602

**Reporte por Clase (Entrenamiento Efectivo)**

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.9522    | 0.8359 | 0.8902   | 262     |
| Miedo    | 0.9372    | 0.8863 | 0.9110   | 387     |
| Sorpresa | 0.6629    | 0.9219 | 0.7712   | 64      |
| Tristeza | 0.7672    | 0.8725 | 0.8165   | 204     |

### Resultados en Validación

- **Accuracy:** 0.7868
- **Precision (macro):** 0.7313
- **Recall (macro):** 0.7283
- **F1-Score (macro):** 0.7270

**AUC ROC (Validación):**

- Macro-promedio: 0.9318
- Alegria: 0.9359
- Miedo: 0.9606
- Sorpresa: 0.9278
- Tristeza: 0.8978

**Reporte por Clase (Validación)**

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.8750    | 0.7500 | 0.8077   | 112     |
| Miedo    | 0.8963    | 0.8802 | 0.8882   | 167     |
| Sorpresa | 0.5556    | 0.5556 | 0.5556   | 27      |
| Tristeza | 0.5981    | 0.7273 | 0.6564   | 88      |

### Resultados en Prueba

- **Accuracy:** 0.8049
- **Precision (macro):** 0.7351
- **Recall (macro):** 0.7765
- **F1-Score (macro):** 0.7477

**AUC ROC (Prueba):**

- Macro-promedio: 0.9345
- Alegria: 0.9509
- Miedo: 0.9762
- Sorpresa: 0.9293
- Tristeza: 0.8746

**Reporte por Clase (Prueba)**

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.9070    | 0.8387 | 0.8715   | 93      |
| Miedo    | 0.9380    | 0.8705 | 0.9030   | 139     |
| Sorpresa | 0.4722    | 0.7391 | 0.5763   | 23      |
| Tristeza | 0.6234    | 0.6575 | 0.6400   | 73      |

### Gráficas Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión RoBERTuito](./cm_conjunta_roberta.png)

**Curvas ROC:**

![Curvas ROC RoBERTuito](./roc_conjunta_roberta.png)

---
