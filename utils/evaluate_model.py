import os
import json
import numpy as np
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from typing import Dict, List, Tuple
from pathlib import Path

class ModelEvaluator:
    def __init__(self, ground_truth_path: str):
        """Initialise l'évaluateur avec le chemin vers le fichier de vérité terrain."""
        self.ground_truth = self._load_ground_truth(ground_truth_path)
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def _load_ground_truth(self, path: str) -> Dict:
        """Charge le fichier de vérité terrain."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def calculate_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calcule l'Intersection over Union entre deux boîtes englobantes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def evaluate_detection(self, predictions: List[Dict], ground_truth: List[Dict], 
                         iou_threshold: float = 0.5) -> Dict:
        """Évalue les performances de détection."""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        ious = []
        
        # Pour chaque prédiction
        for pred in predictions:
            matched = False
            for gt in ground_truth:
                if pred['class_name'] == gt['class_name']:
                    iou = self.calculate_iou(pred['bbox'], gt['bbox'])
                    if iou >= iou_threshold:
                        true_positives += 1
                        ious.append(iou)
                        matched = True
                        break
            if not matched:
                false_positives += 1
                
        false_negatives = len(ground_truth) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        mean_iou = np.mean(ious) if ious else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'mean_iou': mean_iou
        }
    
    def evaluate_annotation(self, predicted_annotation: str, ground_truth_annotation: str) -> Dict:
        """Évalue la qualité de l'annotation textuelle."""
        # Encode les annotations
        pred_embedding = self.sentence_model.encode(predicted_annotation)
        gt_embedding = self.sentence_model.encode(ground_truth_annotation)
        
        # Calcule la similarité cosinus
        similarity = 1 - cosine(pred_embedding, gt_embedding)
        
        return {
            'semantic_similarity': similarity
        }
    
    def evaluate_image(self, image_path: str, predictions: Dict) -> Dict:
        """Évalue une image individuelle."""
        image_name = os.path.basename(image_path)
        if image_name not in self.ground_truth:
            raise ValueError(f"Pas de vérité terrain pour l'image {image_name}")
            
        gt = self.ground_truth[image_name]
        
        detection_metrics = self.evaluate_detection(
            predictions['detections'],
            gt['detections']
        )
        
        annotation_metrics = self.evaluate_annotation(
            predictions['annotation'],
            gt['annotation']
        )
        
        return {
            'detection': detection_metrics,
            'annotation': annotation_metrics
        }
    
    def evaluate_dataset(self, dataset_path: str, predictions: Dict) -> Dict:
        """Évalue l'ensemble du jeu de données."""
        results = {}
        total_metrics = {
            'detection': {
                'precision': [],
                'recall': [],
                'f1_score': [],
                'mean_iou': []
            },
            'annotation': {
                'semantic_similarity': []
            }
        }
        
        for image_name, pred in predictions.items():
            image_path = os.path.join(dataset_path, image_name)
            if os.path.exists(image_path):
                try:
                    metrics = self.evaluate_image(image_path, pred)
                    results[image_name] = metrics
                    
                    # Accumule les métriques
                    for metric_type in ['detection', 'annotation']:
                        for metric, value in metrics[metric_type].items():
                            total_metrics[metric_type][metric].append(value)
                except Exception as e:
                    print(f"Erreur lors de l'évaluation de {image_name}: {str(e)}")
        
        # Calcule les moyennes
        avg_metrics = {
            'detection': {
                metric: np.mean(values) 
                for metric, values in total_metrics['detection'].items()
            },
            'annotation': {
                metric: np.mean(values) 
                for metric, values in total_metrics['annotation'].items()
            }
        }
        
        return {
            'per_image_results': results,
            'average_metrics': avg_metrics
        }

def main():
    # Chemins des fichiers
    ground_truth_path = 'data/ground_truth.json'
    dataset_path = 'data/raw'
    
    # Initialise l'évaluateur
    evaluator = ModelEvaluator(ground_truth_path)
    
    # Charge les prédictions (à remplacer par vos prédictions réelles)
    predictions = {
        "image1.jpg": {
            "detections": [
                {"class_name": "person", "bbox": [110, 210, 310, 410]},
                {"class_name": "car", "bbox": [410, 310, 610, 510]}
            ],
            "annotation": "Une personne se tient debout à gauche de l'image et une voiture rouge est garée à droite. La scène se passe dans une rue de la ville."
        },
        "image2.jpg": {
            "detections": [
                {"class_name": "dog", "bbox": [160, 260, 360, 460]},
                {"class_name": "cat", "bbox": [460, 360, 560, 460]}
            ],
            "annotation": "Un chien brun est assis à gauche et un chat noir se trouve à droite. Les deux animaux semblent se regarder dans un jardin."
        }
    }
    
    # Évalue le jeu de données
    results = evaluator.evaluate_dataset(dataset_path, predictions)
    
    # Affiche les résultats
    print("\nRésultats d'évaluation:")
    print("\nMétriques moyennes:")
    print("\nDétection:")
    for metric, value in results['average_metrics']['detection'].items():
        print(f"{metric}: {value:.3f}")
    
    print("\nAnnotation:")
    for metric, value in results['average_metrics']['annotation'].items():
        print(f"{metric}: {value:.3f}")
    
    print("\nRésultats par image:")
    for image_name, metrics in results['per_image_results'].items():
        print(f"\nImage: {image_name}")
        print("Détection:")
        for metric, value in metrics['detection'].items():
            print(f"  {metric}: {value:.3f}")
        print("Annotation:")
        for metric, value in metrics['annotation'].items():
            print(f"  {metric}: {value:.3f}")

if __name__ == "__main__":
    main() 