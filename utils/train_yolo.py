import os, json, random, shutil, torch, yaml
from pathlib import Path
from typing import Dict, List, Tuple
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict



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


"--- Pour l'entrainement du modèle --- "

class YOLODataset(Dataset):
    def __init__(self, data_dir: str, image_size: int = 640):
        """Initialise le dataset YOLO."""
        self.data_dir = data_dir
        self.image_size = image_size
        self.image_files = []
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Charge les chemins des images
        with open(os.path.join(data_dir, 'train.txt' if 'train' in data_dir else 'val.txt'), 'r') as f:
            self.image_files = [line.strip() for line in f.readlines()]
    
    def __len__(self) -> int:
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_files[idx]
        label_path = image_path.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        
        # Charge l'image
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Charge les labels
        labels = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                labels.append([class_id, x_center, y_center, width, height])
        
        # Convertit les labels en tenseur
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return image, labels

class YOLOv3Trainer:
    def __init__(self, config_path: str):
        """Initialise l'entraîneur YOLOv3."""
        self.config = self._load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.MSELoss()
    
    def _load_config(self, config_path: str) -> Dict:
        """Charge la configuration depuis un fichier YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_model(self) -> nn.Module:
        """Crée le modèle YOLOv3."""
        # Ici, vous devriez implémenter ou importer votre architecture YOLOv3
        # Pour l'exemple, nous utilisons une version simplifiée
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1024, 5, kernel_size=1)  # 5 = 4 coordonnées + 1 classe
        )
        return model.to(self.device)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int):
        """Entraîne le modèle."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Phase d'entraînement
            self.model.train()
            train_loss = 0.0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            
            # Phase de validation
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss /= len(val_loader)
            
            # Affiche les métriques
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Sauvegarde le meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Évalue le modèle sur l'ensemble de test."""
        self.model.eval()
        test_loss = 0.0
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                test_loss += loss.item()
                
                predictions.extend(outputs.cpu().numpy())
                ground_truths.extend(labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        
        return {
            'test_loss': test_loss,
            'predictions': predictions,
            'ground_truths': ground_truths
        }

def main():
    
    # Chemins des fichiers
    ground_truth_path = 'data/ground_truth.json'
    images_dir = 'data/raw/'
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

    # Chemins des fichiers
    config_path = 'config/yolo_config.yaml'
    data_dir = 'data/yolo'
    
    # Crée les datasets
    train_dataset = YOLODataset(os.path.join(data_dir, 'train'))
    val_dataset = YOLODataset(os.path.join(data_dir, 'val'))
    
    # Crée les dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Initialise l'entraîneur
    trainer = YOLOv3Trainer(config_path)
    
    # Entraîne le modèle
    print("Début de l'entraînement...")
    trainer.train(train_loader, val_loader, num_epochs=50)
    
    # Évalue le modèle
    print("\nÉvaluation du modèle...")
    results = trainer.evaluate(val_loader)
    print(f"Test Loss: {results['test_loss']:.4f}")

if __name__ == "__main__":
    main() 