# Utiliser une image Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de dépendances
COPY requirements.txt .

# Installer virtualenv dans le container
RUN pip install --no-cache-dir virtualenv

# Créer l'environnement virtuel
RUN virtualenv venv

# Installer les dépendances dans l'environnement virtuel
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Exposer le port 8080
EXPOSE 8080

# Commande pour lancer Gunicorn avec UvicornWorker via wsgi.py
CMD ["venv/bin/gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "wsgi:app", "--host", "0.0.0.0", "--port", "8080"]