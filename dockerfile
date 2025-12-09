# Dockerfile - GUÍA PASO A PASO

# PASO 1: Imagen base con Python 3.10 (estable y compatible)
FROM python:3.10-slim

# PASO 2: Etiquetas informativas (opcional)
LABEL maintainer="Tu Nombre"
LABEL description="ChatBot Educativo con NLP"
LABEL version="1.0"

# PASO 3: Establecer directorio de trabajo dentro del contenedor
WORKDIR /app

# PASO 4: Instalar dependencias del sistema (si las necesitas)
# RUN apt-get update && apt-get install -y \
#     gcc \
#     g++ \
#     && rm -rf /var/lib/apt/lists/*

# PASO 5: Copiar requirements.txt primero (para cachear dependencias)
COPY requirements.txt .

# PASO 6: Instalar dependencias de Python
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# PASO 7: Descargar datos de NLTK
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# PASO 8: Copiar TODO el código de la aplicación
COPY . .

# PASO 9: Exponer el puerto que usará Streamlit
EXPOSE 8501

# PASO 10: Variable de entorno para el puerto
ENV PORT=8501

# PASO 11: Comando para ejecutar la aplicación
CMD ["streamlit", "run", "chatbot_app.py", "--server.port=8501", "--server.headless=true", "--server.enableCORS=false"]