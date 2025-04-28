import os
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
sys.path.append(str(Path(__file__).parent.parent))

from utils.image_processor import ImageProcessor
from utils.image_annotator import ImageAnnotator

def main():
    # Dossiers à traiter
    directories = [
        "data/raw/",
        #"data/raw/Men",
        #"data/raw/Caps"
    ]
    
    # Créer les dossiers de sortie
    output_dir = Path("data/processed")
    output_dir.mkdir(exist_ok=True)
    
    # Initialiser les processeurs
    image_processor = ImageProcessor()
    image_annotator = ImageAnnotator()
    
    # Traiter chaque répertoire
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Le répertoire {directory} n'existe pas")
            continue
            
        print(f"\nTraitement du répertoire: {directory}")
        
        # 1. Détection des objets avec YOLO
        print("Étape 1: Détection des objets...")
        detection_results = image_processor.process_directory(directory)
        
        # Sauvegarder les détections
        detection_file = output_dir / f"{Path(directory).name}_detections.json"
        with open(detection_file, 'w') as f:
            import json
            json.dump(detection_results, f, indent=4)
        
        # 2. Génération des annotations avec Azure OpenAI
        print("Étape 2: Génération des annotations...")
        annotations = image_annotator.process_directory(directory, detection_results)
        
        # Sauvegarder les annotations
        annotation_file = output_dir / f"{Path(directory).name}_annotations.json"
        image_annotator.save_annotations(annotations, annotation_file)
        
        print(f"Traitement terminé pour {directory}")
        print(f"Détections sauvegardées dans: {detection_file}")
        print(f"Annotations sauvegardées dans: {annotation_file}")

if __name__ == "__main__":
    main() 