import cv2
import numpy as np
import os

# Define paths for each class
class_paths = {
    "Normal Leaf": {
        "input": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_1_Augmentation\01_Normal',
        "output": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\2_Processed_Images\01_normal'
    },
    "Leaf Spot": {
        "input": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_1_Augmentation\02_Leaf_spot',
        "output": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\2_Processed_Images\02_Leaf spot'
    },
    "Mosaic Virus": {
        "input": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_1_Augmentation\03_Mosaic_Virus',
        "output": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\2_Processed_Images\03_Mosaic Virus'
    },
    "Powdery Mildew": { 
        "input": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_1_Augmentation\04_Powdery_Mildew',
        "output": r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\2_Processed_Images\04_Powdery Mildew'
    }
}

# File extensions to check
extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

# Loop over each class
for class_name, paths in class_paths.items():
    os.makedirs(paths['output'], exist_ok=True)

    for i in range(1, 1001):  # 001-1000
        # Generate the base filename with 4 digits
        base_filename = ""
        if class_name == "Normal Leaf":
            base_filename = f"01_Normal_{str(i).zfill(4)}"
        elif class_name == "Leaf Spot":
            base_filename = f"02_Leaf_spot_{str(i).zfill(4)}"
        elif class_name == "Mosaic Virus":
            base_filename = f"03_Mosaic_Virus_{str(i).zfill(4)}"
        elif class_name == "Powdery Mildew":
            base_filename = f"04_Powdery_Mildew_{str(i).zfill(4)}"

        # Check each extension
        loaded = False
        for ext in extensions:
            image_filename = f"{base_filename}{ext}"
            image_path = os.path.join(paths['input'], image_filename)

            # Try loading the image
            print(f"Attempting to load image: {image_path}")  # Debug print
            image = cv2.imread(image_path)
            if image is not None:
                loaded = True
                print(f"Successfully loaded image: {image_path}")

                # Resize and pre-process
                image = cv2.resize(image, (256, 256))

                # Prepare mask with GrabCut
                mask = np.zeros(image.shape[:2], np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)
                rect = (5, 5, image.shape[1] - 10, image.shape[0] - 10)
                cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

                # Refine the mask
                mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # Apply mask and create an RGBA image
                segmented_image = image * mask[:, :, np.newaxis]
                rgba_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2BGRA)
                rgba_image[:, :, 3] = mask * 255  # Set alpha channel based on mask

                # Save processed image
                output_file = os.path.join(paths['output'], f"{base_filename}.png")
                cv2.imwrite(output_file, rgba_image)
                print(f"Saved processed image to: {output_file}")

                break  # Stop once the image is loaded and processed
        
        if not loaded:
            print(f"Error loading image for base filename: {base_filename}")

print("Processing completed.")
