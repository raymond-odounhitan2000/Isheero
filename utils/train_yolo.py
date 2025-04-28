import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
import yaml
from pathlib import Path

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