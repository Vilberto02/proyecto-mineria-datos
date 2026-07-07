# Enlaces a la documentación del proyecto


Resumen de articulos: [Resumen-Articulos](https://docs.google.com/document/d/1t0_zRf1gLioM6RyKwEp5_qsGoJkrrAFue2AN88BZCuE/edit?usp=sharing)

Drive de documentos: [Proyecto-Mineria-G4](https://drive.google.com/drive/folders/1Uljvi6Wt4v6-hpnu_-VaOJBz81WrIGsS?usp=sharing)

Repositorio del proyecto: [Repositorio-Proyecto-Mineria-Datos](https://github.com/Vilberto02/proyecto-mineria-datos)

## Estructura de ramas

En el repositorio de github existen 4 ramas:
- `main`: En esta rama se tiene todo el contenido referente al entrenamiento y validación de los 7 modelos de machine learning utilizando la técnica de validación cruzada para reducir el sobre ajuste de estos modelos, aplicada a los conjuntos de datos preprocesados.
- `preprocesamiento`: Aquí se encuentra alojado el contenido referente al preprocesamiento de los conjuntos de datos, en donde se incluye la limpieza, demojización, entre otros.
- `modelado`: En esta rama se encuentra alojada el contenido referente al entrenamiento de los 7 modelos de machine learning sin la validación cruzada, aplicada a los conjuntos de datos preprocesados.
- `pseudo-etiquetado`: En esta rama se realiza el pseudo-etiquetado de los datos, que consiste en dividir el conjunto de datos en `gold_set` y `silver_set`, el primero contiene conjunto datos con etiquetas ya verificadas y el segundo conjunto es la predicción de etiquetas con el modelo de RoBERTuito.
