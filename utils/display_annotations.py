import json
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

def draw_annotations(image_path, detections, description=None):
    """Dessine les annotations et la description sur l'image"""
    try:
        # Ouvrir l'image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Définir les couleurs pour les différents types d'objets
        colors = {
            'person': 'red',
            'car': 'blue',
            'truck': 'green',
            'bus': 'yellow',
            'motorcycle': 'purple',
            'bicycle': 'orange'
        }
        
        # Dessiner les boîtes englobantes et les étiquettes
        if 'detections' in detections and 'detections' in detections['detections']:
            for det in detections['detections']['detections']:
                x1, y1, x2, y2 = det['bbox']
                color = colors.get(det['class_name'].lower(), 'white')
                # Dessiner un rectangle plus épais
                draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                label = f"{det['class_name']} ({det['confidence']:.2f})"
                # Utiliser une police plus grande pour les étiquettes
                try:
                    font = ImageFont.truetype("arial.ttf", 24)
                except:
                    font = ImageFont.load_default()
                # Dessiner un fond pour le texte
                text_bbox = draw.textbbox((x1, y1 - 30), label, font=font)
                draw.rectangle(text_bbox, fill='black')
                draw.text((x1, y1 - 30), label, fill=color, font=font)
        
        # Ajouter la description si elle est fournie
        if description:
            # Créer une nouvelle image plus grande pour ajouter la description
            width, height = img.size
            new_height = height + 300  # Plus d'espace pour la description
            new_img = Image.new('RGB', (width, new_height), 'white')
            new_img.paste(img, (0, 0))
            
            # Dessiner la description
            draw = ImageDraw.Draw(new_img)
            # Utiliser une police plus grande
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Diviser la description en lignes
            words = description.split()
            lines = []
            current_line = ""
            for word in words:
                if len(current_line + word) < 50:  # Moins de caractères par ligne
                    current_line += word + " "
                else:
                    lines.append(current_line)
                    current_line = word + " "
            if current_line:
                lines.append(current_line)
            
            # Dessiner chaque ligne avec un fond
            y = height + 20
            for line in lines:
                # Dessiner un fond pour le texte
                text_bbox = draw.textbbox((10, y), line, font=font)
                draw.rectangle(text_bbox, fill='white')
                # Dessiner le texte
                draw.text((10, y), line, fill='black', font=font)
                y += 30  # Plus d'espace entre les lignes
            
            return new_img
        
        return img
    except Exception as e:
        print(f"Erreur lors du dessin des annotations: {str(e)}")
        return Image.open(image_path)

def display_annotated_image(image_path, detections, annotation):
    """Affiche l'image avec les annotations et la description"""
    try:
        # Créer une figure avec deux sous-graphiques
        fig = plt.figure(figsize=(12, 10))
        
        # Premier sous-graphique pour l'image
        ax1 = plt.subplot(2, 1, 1)
        annotated_img = draw_annotations(image_path, detections, annotation)
        ax1.imshow(np.array(annotated_img))
        ax1.axis('off')
        ax1.set_title(os.path.basename(image_path), pad=20)
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Erreur lors de l'affichage de l'image annotée: {str(e)}")

def display_annotations(json_file):
    """Affiche les annotations d'un fichier JSON"""
    try:
        abs_path = os.path.abspath(json_file)
        print(f"Lecture du fichier: {abs_path}")
        
        with open(abs_path, 'r', encoding='utf-8') as f:
            annotations = json.load(f)
        
        print(f"\nNombre total d'images annotées: {len(annotations)}\n")
        
        for image_name, data in annotations.items():
            if 'path' in data:
                image_path = data['path']
                print(f"\nAffichage de l'image annotée: {image_name}")
                annotation = data.get('annotation', '')
                display_annotated_image(image_path, data, annotation)
            
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier: {str(e)}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    annotations_file = os.path.join(current_dir, "..", "data", "processed", "Women_annotations.json")
    display_annotations(annotations_file) 