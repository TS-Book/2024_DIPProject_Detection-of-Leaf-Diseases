import os
import cv2
import numpy as np
import joblib
from skimage.feature import hog

# Paths for testing
test_folders = [
    r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\5_Test\01_Normal_Test',
    r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\5_Test\02_LeafSpot_Test',
    r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\5_Test\03_Mosaic Virus_Test',
    r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\5_Test\04_Powdery Mildew_Test'
]
output_folder = r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\6_TestAll'
os.makedirs(output_folder, exist_ok=True)

# Step 1: Rename images and save as PNG
counter = 0
for folder in test_folders:
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            src = os.path.join(folder, filename)
            dst = os.path.join(output_folder, f"{counter}.png")  # Change to .png
            cv2.imwrite(dst, cv2.imread(src))
            counter += 1

# Step 2: Image segmentation, resizing, and background removal
def segment_and_resize_image(image):
    # Resize image for consistent processing
    image_resized = cv2.resize(image, (256, 256))
    height, width = image_resized.shape[:2]
    
    # Initialize mask and GrabCut models
    mask = np.zeros(image_resized.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Define a rectangle for the initial grab cut area
    rect = (5, 5, width - 10, height - 10)
    cv2.grabCut(image_resized, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
    
    # Refine mask to isolate foreground (leaf area)
    mask = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    
    # Optional: further refine mask to remove small gaps or noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Apply mask to create a transparent background image (RGBA)
    segmented_image = image_resized * mask[:, :, np.newaxis]
    rgba_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2BGRA)
    rgba_image[:, :, 3] = mask * 255  # Set alpha channel based on mask
    
    return rgba_image

# Apply segmentation and save the output with transparency
for filename in os.listdir(output_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(output_folder, filename)
        image = cv2.imread(image_path)
        if image is not None:
            segmented_image = segment_and_resize_image(image)
            # Save as PNG to preserve transparency
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            cv2.imwrite(output_path, segmented_image)
            print(f"Processed and saved: {output_path}")

# Step 3: Feature extraction (HOG and HSV color histogram)
def extract_hog_for_color_image(image):
    hog_features = []
    for channel in cv2.split(image):  # Split into B, G, R channels
        features, _ = hog(channel, orientations=8, pixels_per_cell=(16, 16),
                          cells_per_block=(1, 1), visualize=True, channel_axis=None)
        hog_features.append(features)
    return np.concatenate(hog_features)

def calculate_color_histogram(hsv_image, mask):
    color_histogram = []
    mask = mask.astype("uint8")
    for i in range(3):  # H, S, V channels
        hist = cv2.calcHist([hsv_image], [i], mask, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()  # Normalize and flatten
        color_histogram.extend(hist)
    return np.array(color_histogram)

# Step 4: Load the SVM model and scaler, classify images
model_path = r'D:\University\3\3_1\DIP\Mini project\TEST_Plant_Disease_Detection\Data\4_SVM\svm_model.joblib'
scaler_path = r'D:\University\3\3_1\DIP\Mini project\TEST_Plant_Disease_Detection\Data\4_SVM\scaler.joblib'
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
class_labels = ["Normal Leaf", "Leaf Spot", "Mosaic Virus", "Powdery Mildew"]

for filename in os.listdir(output_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(output_folder, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {filename}")
            continue
        
        # Resize and convert to HSV for color hist extraction
        image_resized = cv2.resize(image, (128, 64))
        hsv_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2HSV)
        mask = (hsv_image[:, :, 2] > 20) & (hsv_image[:, :, 1] > 40)
        
        hog_feat = extract_hog_for_color_image(image_resized)
        color_hist = calculate_color_histogram(hsv_image, mask)
        combined_feat = np.concatenate((hog_feat, color_hist)).reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(combined_feat)

        # Predict and display the result
        prediction = model.predict(scaled_features)[0]
        print(f"Image '{filename}' classified as: {class_labels[prediction]}")
