# Selección de modelos de clasificación

Se busca implementar y comparar diferentes familias de algoritmos para la clasificación automática de emociones en publicaciones textuales, con el propósito de identificar el modelo con mejor desempeño sobre el conjunto de datos etiquetado.

La selección de modelos se realizó a partir de la revisión de trabajos relacionadas sobre clasificación de emociones en redes sociales. Se ha identificado 3 categorías para la agrupación de algoritmos de clasificación:

- Modelos clásicos de aprendizaje automático.
- Redes neuronales profundas.
- Transformers.

## 1. Modelos clásicos

Los modelos clásicos utilizan representaciones vectoriales de los textos mediante técnicas como TF-IDF.
Los cuales presentan los siguientes campos de entrada y salida de los algoritmos:

| Entrada       | Salida                        |
| ------------- | ----------------------------- |
| Matriz TF-IDF | Predicción de clase emocional |

### Investigaciones

- [Detection of emotion by text analysis using machine learning - Machová, et al. (2023)](https://www.frontiersin.org/articles/10.3389/fpsyg.2023.1190326/full)
- [Emotion Detection From Social Media Posts - Rahman, et al. (2023)](https://arxiv.org/abs/2302.05610)

### Algoritmos

1. Support Vector Machine (SVM): SVM es un algoritmo supervisado que construye hiperplanos para separar clases en espacios multidimensionales.
2. Naive Bayes: Naive Bayes es un clasificador probabilístico basado en el teorema de Bayes, asumiendo independencia entre características.
3. Árbol de decisión: Los árboles de decisión realizan clasificaciones mediante reglas jerárquicas generadas a partir de los datos.
4. Random Forest: Random Forest combina múltiples árboles de decisión para reducir sobreajuste y mejorar la capacidad predictiva.

## 2. Redes neuronales profundas

Las redes neuronales permiten aprender patrones semánticos y relaciones entre palabras sin depender únicamente de representaciones estadísticas.

Para estos modelos el texto será convertido a secuencias numéricas y posteriormente transformado mediante embeddings.

| Entrada                           | Salida                        |
| --------------------------------- | ----------------------------- |
| texto → tokenización → embeddings | Predicción de clase emocional |

### Investigaciones

- [GO-DEPRESSION: Gannet Optimization Codemix Features Based Coordinate Attention CNN-BiLSTM for Penta Depression Detection - Chitrakala, et al. (2026)](https://doi.org/10.1007/s44196-025-01142-6)
- [A Real-Time Predicting Online Tool for Detection of People's Emotions From Arabic Tweets Based on Big Data Platforms - Abdelhady, et al. (2024)](https://doi.org/10.1186/s40537-024-01035-z)
- [Emotion Detection Face from Social Media Text Using CNN-LSTM Hybrid Model - Kalingwar, et al. (2026)](https://www.jetir.org/papers/JETIRHO06001.pdf)
- [Emotion Detection from Text using CNN, LSTM and Hybrid CNN-LSTM Deep Learning Model - Ansari , et al. (2026)](https://ijsret.com/wp-content/uploads/IJSRET_V12_issue2_598.pdf)

### Algoritmos

1. Convolutional Neural Network (CNN): Las CNN aplican filtros convolucionales sobre secuencias de palabras para extraer patrones locales.
2. Long Short-Term Memory (LSTM): LSTM es una red neuronal recurrente diseñada para procesar secuencias y mantener información relevante mediante mecanismos de memoria.
3. CNN-LSTM: CNN-LSTM combina extracción local de características mediante CNN y análisis secuencial mediante LSTM.

## 3. Modelos basados en Transformers

Los modelos Transformer utilizan mecanismos de atención para generar representaciones contextuales profundas.

| Entrada                          | Salida                        |
| -------------------------------- | ----------------------------- |
| texto → tokenización transformer | Predicción de clase emocional |

### Investigaciones

- [Emotional Similarity Word Embedding Model for Sentiment Analysis - Matsumoto , et al. (2022)](http://www.scielo.org.mx/scielo.php?script=sci_abstract&pid=S1405-55462022000200875&lng=es&nrm=iso&tlng=en)
- [EmoBERTa-X: Advanced Emotion Classifier with Multi-Head Attention and DES for Multilabel Emotion Classification - Labib, et al. (2025)](https://www.mdpi.com/2504-2289/9/2/48)
- [RolEmo: A Role-Aware Commonsense-Augmented Contrastive Learning Framework for Emotion Classification - Abulaish, et al. (2026)](https://www.mdpi.com/2504-4990/8/3/79)
- [A review on emotion detection by using deep learning techniques - Chutia, et al. (2024)](https://doi.org/10.1007/s10462-024-10831-1)
- [Leveraging Distant Supervision and Deep Learning for Twitter Sentiment and Emotion Classification - Kastrati, et al. (2024)](https://doi.org/10.1007/s10844-024-00845-0)

### Algoritmos

1. BETO: BERT en Español, es un modelo preentrenado para tareas de PNL en español
2. RoBERTa: Es una optimización de BERT
3. DistilBERT: BERT reducido

## Métricas de evaluación

- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de confusión
