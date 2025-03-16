import os
import shutil
import numpy as np
import pandas as pd
import cv2
import imagehash
from PIL import Image
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter

base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_path, "dataset", "data")
groups_path = os.path.join(base_path, "dataset", "groups")
csv_path = os.path.join(base_path, "dataset", "groups.csv")

def compute_phash(img_path):
    try:
        img = Image.open(img_path).convert('L').resize((32, 32))
        return imagehash.phash(img)
    except Exception as e:
        print(f"Eroare la citirea imaginii {img_path}: {e}")
        return None

def filter_duplicates(images, threshold=10):
    hashes = {}
    unique_images = []
    duplicate_map = defaultdict(list)
    
    for img_path in images:
        h = compute_phash(img_path)
        if h is None:
            continue 
        
        found_duplicate = None
        for stored_hash, stored_path in hashes.items():
            if h - stored_hash < threshold:
                found_duplicate = stored_path
                break
        if found_duplicate:
            duplicate_map[found_duplicate].append(img_path)
        else:
            hashes[h] = img_path
            unique_images.append(img_path)
    return unique_images, duplicate_map

def extract_shape_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros(7)  
    
    _, thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return np.zeros(7)  

    largest_contour = max(contours, key=cv2.contourArea)
    
    moments = cv2.moments(largest_contour)
    hu_moments = cv2.HuMoments(moments).flatten()

    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)
    
    return hu_moments

def extract_features(img_path):
    return extract_shape_features(img_path)

def cluster_dbscan(features, eps=0.5, min_samples=3):
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(features)

def save_clusters(image_paths, labels):
    os.makedirs(groups_path, exist_ok=True)
    mapping = {}
    
    for img_path, label in zip(image_paths, labels):
        if label == -1:
            continue  

        group_folder = os.path.join(groups_path, f'cluster{label}')
        os.makedirs(group_folder, exist_ok=True)
        shutil.copy(img_path, os.path.join(group_folder, os.path.basename(img_path)))
        mapping[img_path] = label
    
    return mapping

def main():
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.png')]
    print(f"Imagini initiale: {len(image_paths)}")
    
    unique_images, duplicate_map = filter_duplicates(image_paths, threshold=10)
    print(f"Imagini unice dupa eliminarea duplicatelor: {len(unique_images)}")
    
    features = np.array([extract_features(img) for img in unique_images])
    labels = cluster_dbscan(features, eps=0.3, min_samples=2)  
    
    mapping = save_clusters(unique_images, labels)
    
    df = pd.DataFrame([(os.path.splitext(os.path.basename(img))[0], group) for img, group in mapping.items()],
                      columns=['domain', 'cluster'])
    df.to_csv(csv_path, index=False)
    
    print(f"Distributie clustere: {dict(Counter(labels))}")
    print(f"Proces finalizat: {len(image_paths)} imagini analizate, {len(unique_images)} utilizate.")

if __name__ == "__main__":
    main()
