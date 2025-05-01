# Guide d’Installation et d’Utilisation

## 1. Présentation
Le projet **Annotation Automatique d’Images** permet d’analyser et décrire des images via une API REST. Basé sur FastAPI, YOLO et Azure/OpenAI.

## 2. Prérequis
- Python 3.8+
- Docker & Docker Compose (optionnel)
- Clés d’API :
  - AZURE_FORM_RECOGNIZER_ENDPOINT
  - AZURE_FORM_RECOGNIZER_KEY
  - OPENAI_API_KEY

## 3. Installation locale
```bash
git clone <URL_DU_PROJET>
cd <NOM_DU_PROJET>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4. Configuration
Créer un fichier `.env` à la racine et y ajouter :
```dotenv
AZURE_FORM_RECOGNIZER_ENDPOINT=<votre_endpoint>
AZURE_FORM_RECOGNIZER_KEY=<votre_cle>
OPENAI_API_KEY=<votre_cle>
FASTAPI_HOST=0.0.0.0
FASTAPI_PORT=8000
```

## 5. Exécution

### 5.1 En développement
```bash
uvicorn api.main:app --reload --host $FASTAPI_HOST --port $FASTAPI_PORT
```

### 5.2 En production
```bash
gunicorn wsgi:app --workers 4 --bind 0.0.0.0:$FASTAPI_PORT
```

### 5.3 Avec Docker
```bash
docker build -t image-annotator .
docker run -p 8000:8000 --env-file .env image-annotator
```

Optionnel :
```bash
docker-compose up --build
```

## 6. Structure du projet
```
.
├── api/               # Endpoints FastAPI
├── main/              # Scripts principaux
├── models/            # Modèles de détection
├── utils/             # Fonctions d’annotation et d’affichage
├── data/              # Données brutes et traitées
├── config/            # Fichiers de configuration
├── docker/            # Dockerfile et docker-compose.yml
├── wsgi.py            # Point d’entrée production
├── requirements.txt   # Dépendances Python
└── README.md          # Documentation principale
```

## 7. Points d’accès API
- `POST /annotate` : Soumettre une image (paramètre `image`)
- `GET /health/{job_id}` : Vérifier le statut d’un job
- `GET /images/{job_id}` : Récupérer images annotées

## 8. Licence
Ce projet est sous licence **MIT**. Voir `LICENSE` pour plus de détails.
