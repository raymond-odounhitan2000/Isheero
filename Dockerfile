# Utiliser une image Python officielle
FROM python:3.9-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans l'image
COPY requirements.txt .

# Installer pip, virtualenv et mettre à jour pip (si nécessaire)
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir virtualenv

# Créer l'environnement virtuel
RUN python3 -m venv venv

# Installer les dépendances dans l'environnement virtuel
RUN venv/bin/pip install --no-cache-dir -r requirements.txt

# Copier le code source dans l'image
COPY . .

# Exposer le port 8080 pour accéder à l'application
EXPOSE 8080

# Commande pour lancer Gunicorn avec UvicornWorker via wsgi.py
CMD ["/venv/bin/gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "wsgi:app", "--host", "0.0.0.0", "--port", "8080"]

