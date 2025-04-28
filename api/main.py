from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Configuration des logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.image_annotator import ImageAnnotator
    from utils.image_processor import ImageProcessor
    from utils.display_annotations import draw_annotations
except ImportError as e:
    logger.error(f"Erreur d'importation: {str(e)}")
    raise

import base64
from PIL import Image
import io

app = FastAPI(
    title="API d'Annotation d'Images",
    description="API pour l'annotation automatique d'images avec détection d'objets et génération de descriptions",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # remplacer par les origines autorisées lors du deploiement en production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Créer le dossier pour les images annotées
ANNOTATED_IMAGES_DIR = Path("data/annotated")
ANNOTATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Initialisation des processeurs
try:
    image_processor = ImageProcessor()
    image_annotator = ImageAnnotator()
    logger.info("Processeurs initialisés avec succès")
except Exception as e:
    logger.error(f"Erreur lors de l'initialisation des processeurs: {str(e)}")
    raise

@app.post("/annotate")
async def annotate_image(file: UploadFile = File(...)):
    """
    Endpoint pour annoter une image.
    Accepte une image et retourne l'image annotée et sa description.
    """
    temp_path = None
    try:
        logger.info(f"Traitement de l'image: {file.filename}")
        
        # Vérifier le type de fichier
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")

        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        logger.info("Image chargée avec succès")
        
        # Sauvegarder temporairement l'image
        temp_path = f"temp_{file.filename}"
        image.save(temp_path)
        logger.info(f"Image temporaire sauvegardée: {temp_path}")
        
        try:
            # Détection des objets
            logger.info("Début de la détection des objets")
            detections = image_processor.process_image(temp_path)
            logger.info(f"Détections effectuées: {len(detections['detections'])} objets trouvés")
            
            # Génération de l'annotation
            logger.info("Génération de l'annotation")
            annotation = image_annotator.generate_annotation(temp_path, detections)
            logger.info("Annotation générée avec succès")
            
            # Dessiner les annotations sur l'image
            logger.info("Dessin des annotations sur l'image")
            annotated_image = draw_annotations(temp_path, detections, annotation)
            
            # Sauvegarder l'image annotée
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            annotated_filename = f"annotated_{timestamp}_{file.filename}"
            annotated_path = ANNOTATED_IMAGES_DIR / annotated_filename
            annotated_image.save(annotated_path)
            logger.info(f"Image annotée sauvegardée: {annotated_path}")
            
            # Préparer la réponse détaillée
            response_data = {
                "statut": "succès",
                "message": "Image traitée avec succès",
                "résultats": {
                    "chemin_image_annotée": str(annotated_path),
                    "description": annotation,
                    "détections": {
                        "nombre_objets": len(detections['detections']),
                        "objets": [
                            {
                                "classe": det['class_name'],
                                "confiance": f"{det['confidence']:.2f}",
                                "position": {
                                    "x1": det['bbox'][0],
                                    "y1": det['bbox'][1],
                                    "x2": det['bbox'][2],
                                    "y2": det['bbox'][3]
                                }
                            }
                            for det in detections['detections']
                        ]
                    }
                }
            }
            
            return JSONResponse(content=response_data)
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de l'image: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "statut": "erreur",
                    "message": "Erreur lors du traitement de l'image",
                    "détails": str(e)
                }
            )
            
    except Exception as e:
        logger.error(f"Erreur générale: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "statut": "erreur",
                "message": "Erreur serveur",
                "détails": str(e)
            }
        )
        
    finally:
        # Nettoyage
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Fichier temporaire supprimé: {temp_path}")
            except Exception as e:
                logger.error(f"Erreur lors de la suppression du fichier temporaire: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Endpoint pour vérifier l'état de l'API
    """
    return {
        "statut": "en ligne",
        "message": "L'API fonctionne correctement",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 