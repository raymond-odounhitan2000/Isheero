# Projet d'Annotation Automatique d'Images

## 📝 Description du Projet

Ce projet est un système d'annotation automatique d'images qui combine l'intelligence artificielle et la vision par ordinateur pour analyser et décrire automatiquement le contenu des images. Le système est entièrement automatisé et peut être déployé comme un service, permettant l'annotation en temps réel de n'importe quelle image soumise.

## 🎯 Objectifs

- **Automatisation Complète** : Processus d'annotation entièrement automatisé sans intervention humaine
- **Déploiement en Production** : Service disponible 24/7 pour l'annotation d'images
- **Scalabilité** : Capacité à traiter un grand nombre d'images simultanément
- **Fiabilité** : Service robuste avec gestion des erreurs et reprise sur incident
- **Accessibilité** : API simple pour l'intégration dans d'autres systèmes

## 🔄 Processus Automatisé

1. **Soumission d'Image**
   - Upload d'image via API ou interface web
   - Validation automatique du format et de la taille

2. **Traitement Automatique**
   - Détection immédiate des objets
   - Génération des annotations
   - Création des descriptions

3. **Retour des Résultats**
   - Image annotée avec boîtes de détection
   - Description textuelle détaillée
   - Fichier JSON des métadonnées

## 🌐 Déploiement et API

### API REST
```python
# Exemple d'utilisation de l'API
import requests

def annotate_image(image_path):
    url = "https://votre-api.com/annotate"
    files = {'image': open(image_path, 'rb')}
    response = requests.post(url, files=files)
    return response.json()
```

### Points d'Accès
- `POST /annotate` : Annotation d'une image
- `GET /status/{job_id}` : Statut du traitement
- `GET /results/{job_id}` : Récupération des résultats

### Déploiement
1. **Configuration du Serveur**
   ```bash
   # Installation des dépendances
   pip install -r requirements.txt
   
   # Démarrage du serveur
   python main/server.py
   ```

2. **Configuration du Load Balancer**
   - Distribution de la charge entre plusieurs instances
   - Gestion automatique des pics de demande

3. **Monitoring**
   - Surveillance des performances
   - Alertes en cas d'erreur
   - Métriques de traitement

## 🔍 Fonctionnalités Principales

- **Détection d'objets** : Identification automatique des objets dans les images
- **Annotation visuelle** : Affichage des objets détectés avec des boîtes englobantes colorées
- **Description intelligente** : Génération automatique de descriptions détaillées des images
- **Interface visuelle** : Affichage des images avec leurs annotations et descriptions
- **API REST** : Interface programmatique pour l'intégration

## 📊 Résultats Attendus

1. **Annotations Visuelles**
   - Boîtes englobantes autour des objets détectés
   - Couleurs différentes pour chaque type d'objet
   - Niveau de confiance pour chaque détection

2. **Descriptions Textuelles**
   - Analyse détaillée du contenu de l'image
   - Description du contexte et des actions
   - Identification des éléments principaux

3. **Fichiers de Sortie**
   - Images annotées avec les boîtes de détection
   - Fichiers JSON contenant toutes les annotations
   - Rapports de traitement et statistiques

## 🛠️ Technologies Utilisées

- **Azure OpenAI** : Pour la génération des descriptions d'images
- **YOLO (You Only Look Once)** : Pour la détection d'objets
- **Python** : Langage de programmation principal
- **FastAPI** : Framework pour l'API REST
- **Docker** : Containerisation pour le déploiement
- **Bibliothèques Python** :
  - OpenCV : Traitement d'images
  - PIL (Python Imaging Library) : Manipulation d'images
  - Matplotlib : Affichage des images et annotations
  - Azure AI Form Recognizer : Analyse de documents

## 📁 Structure du Projet

```
.
├── api/
│   ├── main.py           # Serveur FastAPI
│   └── endpoints.py      # Points d'accès API
├── data/
│   ├── raw/              # Images originales
│   └── processed/        # Images traitées
├── utils/
│   ├── image_annotator.py    # Gestion des annotations
│   └── display_annotations.py # Affichage des résultats
├── main/
│   ├── process_images.py     # Script principal
│   └── server.py             # Serveur de production
├── models/               # Modèles de détection
├── docker/              # Configuration Docker
└── requirements.txt     # Dépendances du projet
```

## ⚙️ Installation et Déploiement

1. **Installation Locale**
   ```bash
   git clone [URL_DU_PROJET]
   cd [NOM_DU_PROJET]
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Déploiement avec Docker**
   ```bash
   # Construction de l'image
   docker build -t image-annotator .
   
   # Lancement du conteneur
   docker run -p 8000:8000 image-annotator
   ```

3. **Configuration du Load Balancer**
   ```nginx
   upstream annotator {
       server 127.0.0.1:8000;
       server 127.0.0.1:8001;
       server 127.0.0.1:8002;
   }
   ```

## 🎨 Système de Couleurs

- Personnes : 🔴 Rouge
- Voitures : 🔵 Bleu
- Camions : 🟢 Vert
- Bus : 🟡 Jaune
- Motos : 🟣 Violet
- Vélos : 🟠 Orange

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- Azure OpenAI pour l'API de vision
- L'équipe YOLO pour le modèle de détection
- La communauté Python pour les bibliothèques utilisées

## 🚀 Niveaux de Déploiement

### 1. Déploiement Local (Développement)
- **Objectif** : Test et développement
- **Configuration** :
  ```bash
  # Installation locale
  python -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt
  
  # Lancement en mode développement
  python main/server.py --debug
  ```
- **Accès** : `http://localhost:8000`
- **Fonctionnalités** :
  - Mode debug activé
  - Logs détaillés
  - Hot-reload pour le développement

### 2. Déploiement Staging (Pré-production)
- **Objectif** : Tests de validation et QA
- **Configuration** :
  ```bash
  # Construction de l'image Docker
  docker build -t image-annotator:staging .
  
  # Déploiement avec variables d'environnement
  docker run -e ENV=staging \
             -e API_KEY=your_staging_key \
             -p 8000:8000 \
             image-annotator:staging
  ```
- **Accès** : `https://staging.annotator.yourdomain.com`
- **Fonctionnalités** :
  - Environnement de test
  - Données de test
  - Monitoring de base

### 3. Déploiement Production
- **Objectif** : Service en production
- **Architecture** :
  ```
  [Load Balancer]
        ↓
  [Cluster Kubernetes]
  ├── [Pod 1] → Service d'annotation
  ├── [Pod 2] → Service d'annotation
  └── [Pod 3] → Service d'annotation
        ↓
  [Base de données]
  ```
- **Configuration** :
  ```bash
  # Déploiement Kubernetes
  kubectl apply -f k8s/production/
  
  # Configuration du Load Balancer
  kubectl apply -f k8s/loadbalancer.yaml
  ```
- **Accès** : `https://api.annotator.yourdomain.com`
- **Fonctionnalités** :
  - Haute disponibilité
  - Auto-scaling
  - Monitoring avancé
  - Backup automatique
  - Sécurité renforcée

### 4. Déploiement Multi-région
- **Objectif** : Service global avec faible latence
- **Architecture** :
  ```
  [DNS Global]
        ↓
  [Load Balancer Région 1]    [Load Balancer Région 2]
        ↓                              ↓
  [Cluster 1]                 [Cluster 2]
  ├── [Pod 1]                 ├── [Pod 1]
  ├── [Pod 2]                 ├── [Pod 2]
  └── [Pod 3]                 └── [Pod 3]
        ↓                              ↓
  [Base de données 1]         [Base de données 2]
  ```
- **Configuration** :
  ```bash
  # Déploiement multi-région
  terraform apply -var="region=eu-west-1"
  terraform apply -var="region=us-east-1"
  ```
- **Accès** : `https://global.annotator.yourdomain.com`
- **Fonctionnalités** :
  - Distribution géographique
  - Réplication des données
  - Failover automatique
  - CDN intégré

## 🔒 Sécurité et Conformité

### Niveaux de Sécurité
1. **Développement**
   - Authentification basique
   - HTTPS local
   - Logs de debug

2. **Staging**
   - JWT Authentication
   - HTTPS obligatoire
   - Logs sécurisés

3. **Production**
   - OAuth 2.0
   - WAF (Web Application Firewall)
   - DDoS Protection
   - Audit logs
   - Chiffrement des données

### Conformité
- RGPD pour les données personnelles
- ISO 27001 pour la sécurité
- SOC 2 pour la disponibilité
- HIPAA pour les données médicales (si applicable)

## 📈 Monitoring et Maintenance

### Outils de Monitoring
- **Prometheus** : Métriques système
- **Grafana** : Tableaux de bord
- **ELK Stack** : Logs et analyse
- **Sentry** : Gestion des erreurs

### Maintenance
- **Backups** : Quotidien des données
- **Updates** : Mises à jour automatiques
- **Scaling** : Auto-scaling basé sur la charge
- **Alerting** : Notifications en temps réel
