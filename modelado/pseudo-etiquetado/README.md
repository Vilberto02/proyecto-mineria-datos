# Pseudo-etiquetado y filtrado por Clustering

Esta carpeta contiene los scripts necesarios para llevar a cabo el proceso de **pseudo-etiquetado** y **verificación** de las etiquetas generadas en el dataset de emociones.

Dado que contábamos con un gran volumen de datos sin etiquetar o etiquetas no verificadas, y un pequeño volumen de datos muy bien etiquetados, aplicamos esta técnica de aprendizaje semi-supervisado para expandir nuestro set de entrenamiento automáticamente sin inyectar demasiado ruido a los modelos.

## Flujo del Proceso

El proceso completo consta de 3 pasos, cada uno gestionado por un script específico:

### 1. `preparar_datasets_silver_gold.py`

Este es el primer script que se debe ejecutar.

- **Objetivo:** Tomar los datasets originales y separarlos en un _Gold Set_ y un _Silver Target Dataset_.
- **Gold Set:** Mediante un muestreo estratificado, se extraen exactamente 1,000 registros verificados manualmente. Es "estratificado" para garantizar que las clases minoritarias sigan estando representadas de forma justa.
- **Silver Target Dataset:** El resto de los registros verificados y los no verificados se unen en este set (más de 9,000 registros). Estos son los datos que el modelo intentará etiquetar automáticamente.

### 2. `entrenar_robertuito_gold.py`

Este script entrena el modelo "maestro" utilizando únicamente el Gold Set de 1,000 registros.

- **Modelo Base:** Se utiliza `pysentimiento/robertuito-base-uncased`, un transformer especializado en análisis de emociones en textos cortos y redes sociales.
- **Manejo de Desbalance:** Como el Gold Set aún conserva el desbalance natural de las clases, el script calcula dinámicamente los pesos de cada clase (_Class Weights_) y se los pasa a la función de pérdida (`CrossEntropyLoss`). Esto penaliza con más fuerza los errores que el modelo cometa en clases minoritarias como "Sorpresa" o "Asco".
- **Salida:** El modelo entrenado se exporta a la carpeta `../resultados_transformers/robertuito_gold_final`.

### 3. `generar_silver_labels_y_clustering.py`

Este script toma el modelo maestro entrenado y el _Silver Target Dataset_ para generar etiquetas y filtrarlas.

- **Inferencia (Pseudo-Etiquetado):** Pasa todos los registros del Silver Dataset por el modelo entrenado y les asigna una "Silver Label" (Etiqueta de plata).
- **Extracción de Embeddings:** Al mismo tiempo, extrae el vector `[CLS]` de la última capa oculta de RoBERTuito para cada texto. Este vector representa matemáticamente el texto en el espacio semántico.
- **Clustering (HDBSCAN):** Agrupa estos embeddings usando un modelo no supervisado. La intuición es que textos semánticamente similares deben terminar en el mismo grupo.
- **Regla de Filtrado (Class Noise Filtering):** Se analiza cada grupo generado. Si la predicción (Silver Label) que hizo RoBERTuito sobre un registro **coincide** con la etiqueta mayoritaria de su clúster, la etiqueta se considera _confiable_. Si hay un conflicto (por ejemplo, el modelo predijo "Sorpresa" pero el texto cayó en un clúster dominado por "Alegría"), el registro se descarta automáticamente por posible ambigüedad o ruido.
- **Salida:** Los registros que sobreviven al filtro se concatenan nuevamente con el _Gold Set_ para exportar el archivo definitivo: `../datos/dataset_limpio_final.csv`.

## Ejecución

> [!NOTE]
> Dado que se hace _fine-tuning_ de un modelo Transformer (RoBERTuito), **se recomienda fuertemente ejecutar estos scripts en un entorno con GPU** (ej. Google Colab).

Los scripts están configurados para encontrar las rutas dinámicamente, por lo que puedes ejecutarlos de la siguiente manera:

1. `python preparar_datasets_silver_gold.py`
2. `python entrenar_robertuito_gold.py`
3. `python generar_silver_labels_y_clustering.py`

El resultado final será tu `dataset_limpio_final.csv`, el cual ya está configurado para ser consumido tanto por el pipeline de técnicas clásicas (`pipeline.py`) como por el de redes neuronales (`redes_neuronales.py`).
