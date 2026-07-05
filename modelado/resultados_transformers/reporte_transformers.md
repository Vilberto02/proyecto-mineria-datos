# Reporte de evaluación de modelos Transformer

## BETO

### Resultados de Validación Cruzada (5-Folds en Entrenamiento Efectivo)

- **Accuracy Media:** 0.6846
- **Precision Media (macro):** 0.6867
- **Recall Media (macro):** 0.6873
- **F1-Score Media (macro):** 0.6849

## BETO

### Resultados en Entrenamiento Efectivo

- **Accuracy:** 0.8572
- **Precision (macro):** 0.8624
- **Recall (macro):** 0.8593
- **F1-Score (macro):** 0.8567

**AUC ROC (Entrenamiento Efectivo):**

- Macro-promedio: 0.9688
- Alegria: 0.9601\n- Miedo: 0.9757
- Sorpresa: 0.9746\n- Tristeza: 0.9647

**Reporte por Clase (Entrenamiento Efectivo)**

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.9190    | 0.7393 | 0.8194   | 1304    |
| Miedo    | 0.8938    | 0.8850 | 0.8894   | 1122    |
| Sorpresa | 0.7774    | 0.9285 | 0.8462   | 1147    |
| Tristeza | 0.8596    | 0.8842 | 0.8718   | 1503    |

### Resultados en Validación

- **Accuracy:** 0.7169
- **Precision (macro):** 0.7240
- **Recall (macro):** 0.7159
- **F1-Score (macro):** 0.7145

**AUC ROC (Validación):**

- Macro-promedio: 0.9034
- Alegria: 0.8789
- Miedo: 0.9323
- Sorpresa: 0.8907
- Tristeza: 0.9103

**Reporte por Clase (Validación)**

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.7890    | 0.5886 | 0.6742   | 559     |
| Miedo    | 0.7622    | 0.7464 | 0.7542   | 481     |
| Sorpresa | 0.6149    | 0.7398 | 0.6716   | 492     |
| Tristeza | 0.7299    | 0.7888 | 0.7582   | 644     |

### Resultados en Prueba

- **Accuracy:** 0.7259
- **Precision (macro):** 0.7390
- **Recall (macro):** 0.7270
- **F1-Score (macro):** 0.7262

**AUC ROC (Prueba):**

- Macro-promedio: 0.9076
- Alegria: 0.8788
- Miedo: 0.9382
- Sorpresa: 0.8936
- Tristeza: 0.9187

**Reporte por Clase (Prueba)**

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.7994    | 0.5901 | 0.6790   | 466     |
| Miedo    | 0.8259    | 0.7805 | 0.8026   | 401     |
| Sorpresa | 0.5932    | 0.7628 | 0.6674   | 409     |
| Tristeza | 0.7376    | 0.7747 | 0.7557   | 537     |

### Gráficas Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión BETO](./cm_conjunta_beto.png)

**Curvas ROC:**

![Curvas ROC BETO](./roc_conjunta_beto.png)

---

## RoBERTuito

### Resultados de Validación Cruzada (5-Folds en Entrenamiento Efectivo)

- **Accuracy Media:** 0.6966
- **Precision Media (macro):** 0.7015
- **Recall Media (macro):** 0.6974
- **F1-Score Media (macro):** 0.6973

## RoBERTuito

### Resultados en Entrenamiento Efectivo

- **Accuracy:** 0.8649
- **Precision (macro):** 0.8673
- **Recall (macro):** 0.8652
- **F1-Score (macro):** 0.8654

**AUC ROC (Entrenamiento Efectivo):**

- Macro-promedio: 0.9685
- Alegria: 0.9592
- Miedo: 0.9779
- Sorpresa: 0.9746
- Tristeza: 0.9620

**Reporte por Clase (Entrenamiento Efectivo)**
| Clase | Precisión | Recall | F1-Score | Soporte |
|-------|-----------|--------|----------|---------|
| Alegria | 0.8915 | 0.8006 | 0.8436 | 1304 |
| Miedo | 0.8703 | 0.9029 | 0.8863 | 1122 |
| Sorpresa | 0.8696 | 0.8666 | 0.8681 | 1147 |
| Tristeza | 0.8379 | 0.8909 | 0.8636 | 1503 |

### Resultados en Validación

- **Accuracy:** 0.7229
- **Precision (macro):** 0.7228
- **Recall (macro):** 0.7201
- **F1-Score (macro):** 0.7203

**AUC ROC (Validación):**

- Macro-promedio: 0.9077
- Alegria: 0.8909
- Miedo: 0.9346
- Sorpresa: 0.8978
- Tristeza: 0.9063

**Reporte por Clase (Validación)**

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.7495    | 0.6691 | 0.7070   | 559     |
| Miedo    | 0.7495    | 0.7775 | 0.7633   | 481     |
| Sorpresa | 0.6752    | 0.6463 | 0.6604   | 492     |
| Tristeza | 0.7171    | 0.7873 | 0.7506   | 644     |

### Resultados en Prueba

- **Accuracy:** 0.7303
- **Precision (macro):** 0.7325
- **Recall (macro):** 0.7276
- **F1-Score (macro):** 0.7284

**AUC ROC (Prueba):**

- Macro-promedio: 0.9115
- Alegria: 0.8888
- Miedo: 0.9388
- Sorpresa: 0.9046
- Tristeza: 0.9126

**Reporte por Clase (Prueba)**

| Clase    | Precisión | Recall | F1-Score | Soporte |
| -------- | --------- | ------ | -------- | ------- |
| Alegria  | 0.7550    | 0.6545 | 0.7011   | 466     |
| Miedo    | 0.8020    | 0.8080 | 0.8050   | 401     |
| Sorpresa | 0.6617    | 0.6455 | 0.6535   | 409     |
| Tristeza | 0.7112    | 0.8026 | 0.7542   | 537     |

### Gráficas Conjuntas

**Matrices de Confusión:**

![Matrices de Confusión RoBERTuito](./cm_conjunta_roberta.png)

**Curvas ROC:**

![Curvas ROC RoBERTuito](./roc_conjunta_roberta.png)

---
