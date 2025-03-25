FROM jupyter/pyspark-notebook:latest

USER root

# Actualizar e instalar paquetes adicionales
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      curl \
      wget \
      software-properties-common \
      ssh \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configurar variables de entorno para Spark
ENV SPARK_HOME=/usr/local/spark
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip:$PYTHONPATH
ENV PATH=$SPARK_HOME/bin:$PATH

# Instalar paquetes Python adicionales
RUN pip install --no-cache-dir \
    findspark \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    scikit-learn

# Crear directorios para datos y notebooks (usando la ruta por defecto "work")
RUN mkdir -p /home/jovyan/work/data /home/jovyan/work/notebooks

# Ajustar permisos
RUN chown -R jovyan:users /home/jovyan/work

USER jovyan

# Exponer puertos para Jupyter (8888) y Spark UI (4040)
EXPOSE 8888 4040

# Crear un README de ejemplo
RUN echo "# PySpark con JupyterLab\n\nEste contenedor incluye Apache Spark, JupyterLab y librerías básicas (Pandas, NumPy, Matplotlib, etc.).\n\nLos notebooks se almacenan en /home/jovyan/work/notebooks y los datos en /home/jovyan/work/data.\n" > /home/jovyan/work/README.md

# Comando por defecto para iniciar Jupyter sin token ni contraseña
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]