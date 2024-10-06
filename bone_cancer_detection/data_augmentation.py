import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# Define the augmentation pipeline
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.5),
    ], p=0.3),
    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
    ], p=0.3),
    A.OneOf([
        A.IAASharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        A.IAAEmboss(alpha=(0.2, 0.5), strength=(0.5, 1.0), p=0.5),
    ], p=0.3),
    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
    A.Resize(256, 256, always_apply=True),
])

def augment_images(input_dir, output_dir, num_augmented_per_image=5):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in tqdm(os.listdir(input_dir)):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"Failed to read image: {image_path}")
                continue
            
            for i in range(num_augmented_per_image):
                augmented = transform(image=image)
                augmented_image = augmented['image']
                
                output_filename = f"{os.path.splitext(filename)[0]}_aug_{i}.png"
                output_path = os.path.join(output_dir, output_filename)
                
                # Ensure the augmented image is saved as grayscale
                cv2.imwrite(output_path, augmented_image)

# Use the function for each cancer type
cancer_types = ['Chondrosarcoma', 'Ewing Sarcoma', 'Osteosarcoma']

for cancer_type in cancer_types:
    input_dir = f'dataset/{cancer_type}'
    output_dir = f'dataset_augmented/{cancer_type}'
    augment_images(input_dir, output_dir)

print("Data augmentation completed!")