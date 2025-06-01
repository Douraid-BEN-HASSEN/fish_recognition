import cv2
import os
import glob
import random

def plot_yolo_annotations(image_path, label_path, output_dir=None):
    """Visualise les annotations YOLO sur une image
    
    Args:
        image_path: Chemin vers l'image
        label_path: Chemin vers le fichier .txt d'annotations
        output_dir: Dossier pour sauvegarder les résultats (optionnel)
    """
    # Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur: Impossible de charger l'image {image_path}")
        return
    
    h, w = img.shape[:2]
    
    # Lire les annotations
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        # Extraire la bbox (class, x_center, y_center, width, height)
        class_id = int(parts[0])
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h  # Inversion Y
        box_w = float(parts[3]) * w
        box_h = float(parts[4]) * h
        
        # Calculer les coordonnées du rectangle
        x1 = int(x_center - box_w/2)
        y1 = int(y_center - box_h/2)
        x2 = int(x_center + box_w/2)
        y2 = int(y_center + box_h/2)
        
        # Dessiner la bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Extraire les keypoints s'ils existent (format: x y v)
        if len(parts) > 5:
            kps = parts[5:]
            for i in range(0, len(kps), 2):
                kp_x = float(kps[i]) * w
                kp_y = float(kps[i+1]) * h  # Inversion Y

                color = (0, 0, 255)
                cv2.circle(img, (int(kp_x), int(kp_y)), 5, color, -1)
                cv2.putText(img, str(i//3), (int(kp_x), int(kp_y)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    
    # Afficher ou sauvegarder
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        cv2.imwrite(output_path, img)
        print(f"Résultat sauvegardé: {output_path}")
    else:
        img = cv2.resize(img, (int(3840*0.35), int(2160*0.35)))
        cv2.imshow('YOLO Annotations', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def test_dataset(images_dir, labels_dir, sample_size=5):
    """Teste un échantillon aléatoire du dataset
    
    Args:
        images_dir: Dossier contenant les images
        labels_dir: Dossier contenant les annotations .txt
        sample_size: Nombre d'images à tester
    """
    # Lister toutes les images
    image_files = glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png'))
    random.shuffle(image_files)
    
    # Tester un échantillon
    for img_path in image_files[:sample_size]:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        
        if os.path.exists(label_path):
            print(f"\nVisualisation de: {os.path.basename(img_path)}")
            plot_yolo_annotations(img_path, label_path)
        else:
            print(f"Annotation manquante pour: {os.path.basename(img_path)}")

if __name__ == "__main__":
    IMAGES_DIR = "./datasets/sardine_pose_dataset/train/images"  # Dossier contenant les images
    LABELS_DIR = "./datasets/sardine_pose_dataset/train/labels"  # Dossier contenant les annotations .txt
    OUTPUT_DIR = "./visualizations"          # Dossier pour sauvegarder les résultats
    
    # Tester 5 images aléatoires
    test_dataset(IMAGES_DIR, LABELS_DIR, sample_size=50)