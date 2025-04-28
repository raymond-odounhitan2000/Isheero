import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

class YOLODataPreparator:
    def __init__(self, ground_truth_path: str, images_dir: str, output_dir: str):
        """Initialise le préparateur de données YOLO."""
        self.ground_truth_path = ground_truth_path
        self.images_dir = images_dir
        self.output_dir = output_dir
        self.classes = set()
        
    def load_ground_truth(self) -> Dict:
        """Charge le fichier de vérité terrain."""
        with open(self.ground_truth_path, 'r') as f:
            return json.load(f)
    
    def get_all_classes(self, ground_truth: Dict) -> List[str]:
        """Extrait toutes les classes uniques des annotations."""
        classes = set()
        for image_data in ground_truth.values():
            for detection in image_data['detections']:
                classes.add(detection['class_name'])
        return sorted(list(classes))
    
    def convert_bbox_to_yolo(self, bbox: List[int], img_width: int, img_height: int) -> List[float]:
        """Convertit les coordonnées de la boîte englobante au format YOLO."""
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return [x_center, y_center, width, height]
    
    def prepare_data(self, train_ratio: float = 0.8):
        """Prépare les données au format YOLO et les divise en ensembles d'entraînement et de test."""
        # Crée les répertoires nécessaires
        train_dir = os.path.join(self.output_dir, 'train')
        val_dir = os.path.join(self.output_dir, 'val')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Charge les données
        ground_truth = self.load_ground_truth()
        self.classes = self.get_all_classes(ground_truth)
        
        # Crée le fichier classes.txt
        with open(os.path.join(self.output_dir, 'classes.txt'), 'w') as f:
            for i, class_name in enumerate(self.classes):
                f.write(f"{class_name}\n")
        
        # Divise les données
        image_names = list(ground_truth.keys())
        random.shuffle(image_names)
        split_idx = int(len(image_names) * train_ratio)
        train_images = image_names[:split_idx]
        val_images = image_names[split_idx:]
        
        # Prépare les données d'entraînement
        for image_name in train_images:
            self._prepare_image(image_name, ground_truth[image_name], train_dir)
        
        # Prépare les données de validation
        for image_name in val_images:
            self._prepare_image(image_name, ground_truth[image_name], val_dir)
        
        # Crée les fichiers train.txt et val.txt
        self._create_data_files(train_dir, val_dir)
        
        return len(train_images), len(val_images)
    
    def _prepare_image(self, image_name: str, image_data: Dict, output_dir: str):
        """Prépare une image et ses annotations au format YOLO."""
        # Copie l'image
        src_image_path = os.path.join(self.images_dir, image_name)
        dst_image_path = os.path.join(output_dir, image_name)
        shutil.copy2(src_image_path, dst_image_path)
        
        # Crée le fichier d'annotations
        annotation_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + '.txt')
        with open(annotation_path, 'w') as f:
            for detection in image_data['detections']:
                class_idx = self.classes.index(detection['class_name'])
                bbox = detection['bbox']
                yolo_bbox = self.convert_bbox_to_yolo(bbox, 640, 640)  # Supposons une taille d'image de 640x640
                f.write(f"{class_idx} {' '.join(map(str, yolo_bbox))}\n")
    
    def _create_data_files(self, train_dir: str, val_dir: str):
        """Crée les fichiers train.txt et val.txt avec les chemins des images."""
        with open(os.path.join(self.output_dir, 'train.txt'), 'w') as f:
            for image_name in os.listdir(train_dir):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    f.write(f"{os.path.join(train_dir, image_name)}\n")
        
        with open(os.path.join(self.output_dir, 'val.txt'), 'w') as f:
            for image_name in os.listdir(val_dir):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    f.write(f"{os.path.join(val_dir, image_name)}\n")

def main():
    # Chemins des fichiers
    ground_truth_path = 'data/ground_truth.json'
    images_dir = 'data/raw'
    output_dir = 'data/yolo'
    
    # Initialise le préparateur
    preparator = YOLODataPreparator(ground_truth_path, images_dir, output_dir)
    
    # Prépare les données
    train_count, val_count = preparator.prepare_data()
    
    print(f"\nPréparation des données terminée:")
    print(f"- Nombre d'images d'entraînement: {train_count}")
    print(f"- Nombre d'images de validation: {val_count}")
    print(f"- Classes détectées: {', '.join(preparator.classes)}")
    print(f"\nLes données sont prêtes dans le répertoire: {output_dir}")

if __name__ == "__main__":
    main() 