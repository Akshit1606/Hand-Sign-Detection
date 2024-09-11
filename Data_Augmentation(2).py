import os
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
COMBINED_DATA_DIR = './combined_data'
AUGMENTED_DATA_DIR = './augmented_data'

# Create augmented data directories if they don't exist
if not os.path.exists(AUGMENTED_DATA_DIR):
    os.makedirs(AUGMENTED_DATA_DIR)

# Data augmentation configuration
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Loop through each sign directory
for sign_dir in os.listdir(COMBINED_DATA_DIR):
    sign_path = os.path.join(COMBINED_DATA_DIR, sign_dir)
    augmented_sign_path = os.path.join(AUGMENTED_DATA_DIR, sign_dir)
    
    if not os.path.exists(augmented_sign_path):
        os.makedirs(augmented_sign_path)
    
    for img_file in os.listdir(sign_path):
        img_path = os.path.join(sign_path, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Skipping invalid or corrupted image: {img_path}")
            continue
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_expanded = img_rgb.reshape((1,) + img_rgb.shape)
        
        # Generate and save augmented images
        aug_iter = datagen.flow(img_expanded, batch_size=1)
        for i in range(3):  # Generate 3 augmented images per original image
            aug_img = next(aug_iter)[0].astype('uint8')
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            aug_img_path = os.path.join(augmented_sign_path, f"{os.path.splitext(img_file)[0]}_aug{i}.jpg")
            cv2.imwrite(aug_img_path, aug_img_bgr)

print("Data augmentation complete!")
