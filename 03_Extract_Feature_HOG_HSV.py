import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Function to extract HOG features for each color channel and combine them
def extract_hog_for_color_image(image):
    hog_features = []
    for channel in cv2.split(image):  # Split the image into its B, G, R channels
        features, _ = hog(channel, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=True, channel_axis=None)
        hog_features.append(features)
    return np.concatenate(hog_features)

# Function to calculate Color histogram (HSV) and concatenate with HOG features
def calculate_color_histogram(hsv_image, mask):
    color_histogram = []
    for i in range(3):  # For each channel: H, S, V
        hist = cv2.calcHist([hsv_image], [i], mask.astype(np.uint8), [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten histogram
        color_histogram.extend(hist)
    return np.array(color_histogram)

# Function to extract combined HOG and HSV features for each image
def extract_combined_features(input_folder):
    combined_features = []
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {filename}. Skipping this file.")
                continue
            
            image_resized = cv2.resize(image, (128, 64))
            hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)

            # Create mask
            mask = (hsv_image[:, :, 2] > 20) & (hsv_image[:, :, 1] > 40)
            filtered_image = cv2.bitwise_and(image_resized, image_resized, mask=mask.astype(np.uint8))

            # Extract HOG features
            hog_feat = extract_hog_for_color_image(filtered_image)

            # Calculate HSV histogram
            color_hist = calculate_color_histogram(hsv_image, mask)

            # Combine HOG and Color histogram features
            combined_feat = np.concatenate((hog_feat, color_hist))
            combined_features.append(combined_feat)

    return np.vstack(combined_features)

# Paths and settings
classes = {
    "Normal": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\2_Processed_Images\01_normal',
    "Leaf Spot": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\2_Processed_Images\02_Leaf spot',
    "Mosaic Virus": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\2_Processed_Images\03_Mosaic Virus',
    "Powdery Mildew": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\2_Processed_Images\04_Powdery Mildew'
}

output_hog_folder = r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\3_Hog'

# Loop through each class and save combined features
for idx, (class_name, input_folder) in enumerate(classes.items()):
    # Create a specific output folder for each class
    class_output_folder = os.path.join(output_hog_folder, f"{str(idx+1).zfill(2)}_{class_name.replace(' ', '_')}")
    os.makedirs(class_output_folder, exist_ok=True)

    # Extract combined features for the current class
    combined_features = extract_combined_features(input_folder)

    # Save the combined features to an .npy file for each class
    output_file = os.path.join(class_output_folder, f'HOG_ColorHist_features_{class_name.replace(" ", "_")}.npy')
    np.save(output_file, combined_features)

    print(f"Saved combined HOG and Color histogram features for {class_name} in {output_file}")

print("HOG and Color histogram feature extraction and saving completed.")
