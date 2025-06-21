# Utilise une image Python légère compatible multi-architecture
FROM python:3.11-slim

# Variables d'environnement pour optimiser Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Ajout du chemin Python pour le projet
ENV PYTHONPATH=/app

# Installe les dépendances système nécessaires pour votre projet RAG
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crée un dossier de travail dans le container
WORKDIR /app

# Copie d'abord les fichiers de configuration Poetry pour optimiser le cache Docker
COPY pyproject.toml poetry.lock* ./

# Installe Poetry avec la version spécifiée
RUN pip install poetry==1.8.3

# Configure Poetry pour ne pas créer d'environnement virtuel (optimal pour Docker)
RUN poetry config virtualenvs.create false

# Installe les dépendances via Poetry
# --no-dev: pas de dépendances de développement
# --no-interaction: mode non-interactif
# --no-ansi: pas de couleurs dans les logs
RUN poetry install --no-dev --no-interaction --no-ansi

# Copie le reste des fichiers du projet
COPY . .

# Crée les dossiers nécessaires pour les données et la configuration
RUN mkdir -p /app/data /app/config /app/core

# Expose le port Streamlit standard
EXPOSE 8501

# Configuration Streamlit pour Docker
# --server.address=0.0.0.0 : écoute sur toutes les interfaces
# --server.port=8501 : port standard Streamlit
# --server.headless=true : mode headless pour Docker
# --server.enableCORS=false : désactive CORS pour Docker
# --server.enableXsrfProtection=false : désactive XSRF pour Docker
CMD ["poetry", "run", "streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]