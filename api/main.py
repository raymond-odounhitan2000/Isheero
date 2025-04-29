from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, RedirectResponse, HTMLResponse
import os, sys, logging, shutil
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from datetime import datetime
from fastapi.templating import Jinja2Templates

# Chemin vers ton dossier templates
templates = Jinja2Templates(directory="web_interface")


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

import base64,io
from PIL import Image
# Initialisation de l'application FastAPI
app = FastAPI(
    title="API d'Annotation d'Images",
    description="API pour l'annotation automatique d'images avec détection d'objets et génération de descriptions",
    version="1.0.0"
)

# Servir les fichiers statiques si besoin (css, js, images)
app.mount("/static", StaticFiles(directory="api/static"), name="static")
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


# ROUTE PRINCIPALE (Page HTML directement)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Affiche la page d'accueil avec possibilité d'uploader une image.
    """
    images_dir = Path("/data/raw/women")
    image_files = list(images_dir.glob("*.[jp][pn]g"))[:6]
    images_data = []

    for image_file in image_files:
        with open(image_file, "rb") as img:
            encoded_str = base64.b64encode(img.read()).decode('utf-8')
            images_data.append({
                "filename": image_file.name,
                "base64": f"data:image/jpeg;base64,{encoded_str}"
            })

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "images": images_data}
    )
    
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


# Route pour afficher l'image annotée après traitement
@app.get("/images")
async def display_image(request: Request):
    """
    Affiche l'image annotée après traitement.
    """
    # Récupére le chemin de l'image annotée depuis la query string
    image_path = request.query_params.get("image_path")
    
    # Vérifier si l'image annotée existe dans le dossier /static/annotated
    if image_path and os.path.exists(f"api/static/{image_path}"):
        image_url = f"/static/{image_path}"  # Créer l'URL pour l'image
        return templates.TemplateResponse("images_display.html", {"request": request, "image_url": image_url})
    
    # Si aucune image n'est trouvée, rediriger vers la page d'accueil
    return RedirectResponse("/", status_code=303)

# ROUTE POUR LE TRAITEMENT DU FORMULAIRE
@app.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...)):
    """
    Permet de poster une image via le formulaire HTML et de la traiter.
    """
    temp_path = None
    try:
        logger.info(f"Traitement de l'image via formulaire: {file.filename}")
        
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Le fichier doit être une image")

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        temp_path = f"temp_{file.filename}"
        image.save(temp_path)
        
        detections = image_processor.process_image(temp_path)
        annotation = image_annotator.generate_annotation(temp_path, detections)
        annotated_image = draw_annotations(temp_path, detections, annotation)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        annotated_filename = f"annotated_{timestamp}_{file.filename}"
        annotated_path = ANNOTATED_IMAGES_DIR / annotated_filename
        annotated_image.save(annotated_path)
        
        logger.info(f"Image annotée sauvegardée : {annotated_path}")
        image_url = f"/static/annotated/{os.path.basename(annotated_path)}"
        static_annotated_dir = os.path.join(os.getcwd(), 'api', 'static', 'annotated')
        os.makedirs(static_annotated_dir, exist_ok=True)
        shutil.copy(annotated_path, os.path.join(static_annotated_dir, os.path.basename(annotated_path)))
        return RedirectResponse(f"/images?image_path={annotated_path}", status_code=303)

    except Exception as e:
        logger.error(f"Erreur lors de l'upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    
@app.get("/health")
async def health_check():
    return {
        "statut": "en ligne",
        "message": "L'API fonctionne correctement",
        "version": "1.0.0"
    }



# Lance l'application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
