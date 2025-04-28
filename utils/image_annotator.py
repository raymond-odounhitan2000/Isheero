import os
import json
import base64
import io
from PIL import Image
from openai import AzureOpenAI
from dotenv import load_dotenv

class ImageAnnotator:
    def __init__(self):
        """Initialise l'annotateur d'images avec Azure OpenAI"""
        load_dotenv()
        
        self.client = AzureOpenAI(
            api_key=os.getenv("API_KEY"),
            api_version=os.getenv("VERSION_API"),
            azure_endpoint=os.getenv("ENDPOINT")
        )

    def _prepare_image(self, image_path):
        """Prépare l'image pour l'envoi à Azure OpenAI"""
        with Image.open(image_path) as img:
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()

    def generate_annotation(self, image_path, detections):
        """
        Génère une annotation détaillée pour une image
        Args:
            image_path: Chemin vers l'image
            detections: Résultats de la détection YOLO
        Returns:
            str: Annotation générée
        """
        try:
            # Préparer l'image
            img_str = self._prepare_image(image_path)
            
            # Préparer le contexte des détections
            detection_context = ""
            if detections and 'detections' in detections:
                detection_context = "Objets détectés :\n"
                for det in detections['detections']:
                    detection_context += f"- {det['class_name']} (confiance: {det['confidence']:.2f})\n"

            # Générer l'annotation
            response = self.client.chat.completions.create(
                model=os.getenv("NOM_DEPLOIEMENT"),
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert en analyse d'images. Fournis moi une description détaillée de l'image en tenant compte des objets détectés."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Analyse cette image et fournit une description détaillée. {detection_context}"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_str}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Erreur lors de la génération de l'annotation pour {image_path}: {str(e)}")
            return None

    def process_directory(self, directory_path, detection_results):
        """
        Traite toutes les images d'un répertoire
        Args:
            directory_path: Chemin vers le répertoire
            detection_results: Résultats de la détection YOLO
        Returns:
            dict: Annotations pour chaque image
        """
        annotations = {}
        
        # Créer un mapping des chemins d'images vers leurs détections
        detection_map = {det['image_path']: det for det in detection_results}
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    print(f"Annotation de l'image: {image_path}")
                    
                    # Récupérer les détections pour cette image
                    detections = detection_map.get(image_path, {})
                    
                    # Générer l'annotation
                    annotation = self.generate_annotation(image_path, detections)
                    if annotation:
                        annotations[file] = {
                            "path": image_path,
                            "detections": detections,
                            "annotation": annotation
                        }
        
        return annotations

    def save_annotations(self, annotations, output_file):
        """
        Sauvegarde les annotations dans un fichier JSON
        Args:
            annotations: Dictionnaire des annotations
            output_file: Chemin du fichier de sortie
        """
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=4) 