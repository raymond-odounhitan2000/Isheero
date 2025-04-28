import os
import cv2
import numpy as np
from pathlib import Path
import urllib.request

class ImageProcessor:
    def __init__(self, model_name="yolov3-tiny"):
        """
        Initialise le processeur d'images avec OpenCV DNN
        Args:
            model_name: Nom du modèle à utiliser (yolov3-tiny ou yolov3)
        """
        # Télécharger les fichiers du modèle si nécessaire
        self.model_path = Path("models")
        self.model_path.mkdir(exist_ok=True)
        
        # Configuration des chemins
        if model_name == "yolov3-tiny":
            self.config_file = self.model_path / "yolov3-tiny.cfg"
            self.weights_file = self.model_path / "yolov3-tiny.weights"
            self.classes_file = self.model_path / "coco.names"
            
            # URLs des fichiers
            config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg"
            weights_url = "https://pjreddie.com/media/files/yolov3-tiny.weights"
            classes_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        else:
            self.config_file = self.model_path / "yolov3.cfg"
            self.weights_file = self.model_path / "yolov3.weights"
            self.classes_file = self.model_path / "coco.names"
            
            # URLs des fichiers
            config_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg"
            weights_url = "https://pjreddie.com/media/files/yolov3.weights"
            classes_url = "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
        
        # Télécharger les fichiers si nécessaire
        if not self.config_file.exists():
            print(f"Téléchargement de {config_url}...")
            urllib.request.urlretrieve(config_url, self.config_file)
        
        if not self.weights_file.exists():
            print(f"Téléchargement de {weights_url}...")
            urllib.request.urlretrieve(weights_url, self.weights_file)
        
        if not self.classes_file.exists():
            print(f"Téléchargement de {classes_url}...")
            urllib.request.urlretrieve(classes_url, self.classes_file)
        
        # Charger les classes
        with open(self.classes_file, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # Charger le modèle
        self.net = cv2.dnn.readNetFromDarknet(str(self.config_file), str(self.weights_file))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Obtenir les noms des couches de sortie
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i-1] for i in self.net.getUnconnectedOutLayers()]

    def process_image(self, image_path):
        """
        Traite une image avec YOLO via OpenCV DNN et retourne les détections
        Args:
            image_path: Chemin vers l'image à traiter
        Returns:
            dict: Résultats de la détection
        """
        try:
            # Charger l'image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Impossible de charger l'image: {image_path}")
            
            height, width = img.shape[:2]
            
            # Préparer l'image pour le réseau
            blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
            
            # Effectuer la détection
            self.net.setInput(blob)
            layer_outputs = self.net.forward(self.output_layers)
            
            # Traiter les détections
            boxes = []
            confidences = []
            class_ids = []
            
            for output in layer_outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.5:  # Seuil de confiance
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Appliquer la suppression non-maximale
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    detections.append({
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'confidence': float(confidences[i]),
                        'class': int(class_ids[i]),
                        'class_name': self.classes[class_ids[i]]
                    })
            
            return {
                'image_path': image_path,
                'detections': detections,
                'original_size': (height, width)
            }
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {image_path}: {str(e)}")
            return None

    def process_directory(self, directory_path):
        """
        Traite toutes les images d'un répertoire
        Args:
            directory_path: Chemin vers le répertoire contenant les images
        Returns:
            list: Liste des résultats de détection pour chaque image
        """
        results = []
        for root, _, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(root, file)
                    print(f"Traitement de l'image: {image_path}")
                    result = self.process_image(image_path)
                    if result:
                        results.append(result)
        return results

    def draw_detections(self, image_path, detections, output_path=None):
        """
        Dessine les détections sur l'image
        Args:
            image_path: Chemin vers l'image originale
            detections: Résultats de la détection
            output_path: Chemin où sauvegarder l'image annotée
        Returns:
            np.ndarray: Image avec les détections dessinées
        """
        img = cv2.imread(image_path)
        for det in detections['detections']:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            cls_name = det['class_name']
            
            # Dessiner le rectangle
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Ajouter le texte
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label, (int(x1), int(y1)-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, img)
        
        return img 