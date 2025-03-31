# ----------------------------------------------------------------
# Dockerfile actualizado para PySpark, JupyterLab y Dash
# ----------------------------------------------------------------
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

# Configurar variables de entorno de Spark
ENV SPARK_HOME=/usr/local/spark
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip:$PYTHONPATH
ENV PATH=$SPARK_HOME/bin:$PATH

# Instalar módulos Python necesarios para análisis, Dash y manejo de Parquet
RUN pip install --no-cache-dir \
    py4j \
    pyarrow \
    findspark \
    matplotlib \
    seaborn \
    pandas \
    numpy \
    scikit-learn \
    plotly \
    dash==2.14.1 \
    dash-table \
    scipy

# Crear directorios para datos y notebooks
RUN mkdir -p /home/jovyan/work/data /home/jovyan/work/notebooks
RUN chown -R jovyan:users /home/jovyan/work

USER jovyan

# Exponer los puertos: Jupyter (8888), Spark UI (4040) y Dash (8050)
EXPOSE 8888 4040 8050

# Escribir un README simple
RUN echo "# PySpark con JupyterLab y Dashboard en Dash" > /home/jovyan/work/README.md

# Comando por defecto para iniciar Jupyter sin token ni contraseña
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]# ----------------------------------------------------------------
  # Dockerfile actualizado para PySpark, JupyterLab y Dash
  # ----------------------------------------------------------------
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
  
  # Configurar variables de entorno de Spark
  ENV SPARK_HOME=/usr/local/spark
  ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip
  ENV PATH=$SPARK_HOME/bin:$PATH
  
  # Instalar módulos Python necesarios para análisis, Dash y manejo de Parquet
  RUN pip install --no-cache-dir \
      py4j \
      pyarrow \
      findspark \
      matplotlib \
      seaborn \
      pandas \
      numpy \
      scikit-learn \
      plotly \
      dash==2.14.1 \
      dash-table \
      scipy
  
  # Crear directorios para datos y notebooks
  RUN mkdir -p /home/jovyan/work/data /home/jovyan/work/notebooks
  RUN chown -R jovyan:users /home/jovyan/work
  
  USER jovyan
  
  # Exponer los puertos: Jupyter (8888), Spark UI (4040) y Dash (8050)
  EXPOSE 8888 4040 8050
  
  # Escribir un README simple
  RUN echo "# PySpark con JupyterLab y Dashboard en Dash" > /home/jovyan/work/README.md
  
  # Comando por defecto para iniciar Jupyter sin token ni contraseña
  CMD ["start-notebook.py", "--NotebookApp.token=''", "--NotebookApp.password=''"]