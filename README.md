# Big-Data-Workshop-Spark

Este proyecto levanta un entorno Docker con Jupyter Notebook y PySpark.

## Estructura del directorio

- **Dockerfile**: Define la imagen personalizada basada en `jupyter/pyspark-notebook`.
- **docker-compose.yml**: Orquesta el contenedor y monta volúmenes para notebooks y datos.
- **notebooks/**: Directorio donde colocarás tus notebooks (por ejemplo, `mi_notebook.ipynb`).
- **data/**: Directorio para almacenar tus datasets (como el dataset de Kaggle).
- **README.md**: Este archivo de documentación.

## Cómo usar

1. Crea la estructura de directorios como se muestra arriba.
2. Coloca tus notebooks en el directorio `notebooks` y tus datos en `data`.
3. Desde la raíz del proyecto, levanta el entorno con:
   ```bash
   docker-compose up -d
4. levantar la UI con el comando $ bash 
(base) jovyan@01cddcc11d8b:~$ python /home/jovyan/work/dashboard.py
en el bash del contenedor correspondiente.