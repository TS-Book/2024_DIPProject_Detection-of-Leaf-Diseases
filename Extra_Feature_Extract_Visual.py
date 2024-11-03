# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:50:06 2024

@author: thana
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 04:13:54 2024

@author: thana
"""

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

# Function to extract HOG features and plot them in a combined figure for each class
def extract_and_plot_hog_features(input_folder, class_name, output_folder):
    hog_features = []
    filenames = []

    plt.figure(figsize=(15, 15))  # Adjust figure size if needed

    # Loop through files to extract HOG features and Color histograms
    for idx, filename in enumerate(os.listdir(input_folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):  # Filter image files
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
            hog_feat_normalized = cv2.normalize(hog_feat, None, norm_type=cv2.NORM_L2)  # Normalize HOG features

            # Extract Color histogram features (HSV)
            color_hist = calculate_color_histogram(hsv_image, mask)
            color_hist_normalized = cv2.normalize(color_hist, None, norm_type=cv2.NORM_L2)  # Normalize Color histogram

            # Concatenate normalized HOG and Color histogram features
            combined_features = np.concatenate((hog_feat_normalized, color_hist_normalized))
            hog_features.append(combined_features)
            filenames.append(filename)

            # Visualize Original Image
            plt.subplot(3, 3, idx * 3 + 1)
            plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
            plt.title(f'Original: {filename}')
            plt.axis('off')

            # Visualize HOG Features
            hog_image = hog(filtered_image, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, channel_axis=-1)[1]
            hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
            plt.subplot(3, 3, idx * 3 + 2)
            plt.imshow(hog_image_rescaled, cmap='gray')
            plt.title(f'HOG: {filename}')
            plt.axis('off')

            # Visualize HSV Histogram for each individual image
            plt.subplot(3, 3, idx * 3 + 3)
            hsv_labels = ['Hue', 'Saturation', 'Value']
            colors = ['b', 'g', 'r']
            for j, label in enumerate(hsv_labels):
                hist = cv2.calcHist([hsv_image], [j], mask.astype(np.uint8), [256], [0, 256])
                hist = hist / hist.sum()  # Normalize to match the individual-style histogram
                plt.plot(hist, color=colors[j], alpha=0.7, label=label)
                plt.title(f'HSV Histogram: {filename}')
                plt.xlim([0, 256])
                plt.legend()
                plt.axis('on')


    # Save the combined plot for each class
    combined_plot_path = os.path.join(output_folder, f'HOG_Histogram_{class_name}.png')
    plt.tight_layout()
    plt.savefig(combined_plot_path)
    plt.show()

    # Save combined HOG and Color histogram features
    hog_features_array = np.vstack(hog_features)  # ใช้ np.vstack เพื่อให้แน่ใจว่ามิติเป็น 2D
    output_file = os.path.join(output_folder, f'HOG_ColorHist_features_{class_name}.npy')
    np.save(output_file, hog_features_array)

# Paths
classes = {
    "Normal": r'D:\University\3\3_1\DIP\Mini project\TEST_Plant_Disease_Detection\Data\2_Processed_Images\01_normal',
    "Leaf Spot": r'D:\University\3\3_1\DIP\Mini project\TEST_Plant_Disease_Detection\Data\2_Processed_Images\02_Leaf spot',
    "Mosaic Virus": r'D:\University\3\3_1\DIP\Mini project\TEST_Plant_Disease_Detection\Data\2_Processed_Images\03_Mosaic Virus',
    "Powdery Mildew": r'D:\University\3\3_1\DIP\Mini project\TEST_Plant_Disease_Detection\Data\2_Processed_Images\04_Powdery Mildew'
}

output_hog_folder = r'D:\University\3\3_1\DIP\Mini project\TEST_Plant_Disease_Detection\Data\3_Hog'

# Loop through each class
for idx, (class_name, input_folder) in enumerate(classes.items()):
    class_output_folder = os.path.join(output_hog_folder, f"{str(idx+1).zfill(2)}_{class_name.replace(' ', '_')}")
    os.makedirs(class_output_folder, exist_ok=True)
    extract_and_plot_hog_features(input_folder, class_name, class_output_folder)

print("HOG and Color histogram feature extraction and plotting completed.")
