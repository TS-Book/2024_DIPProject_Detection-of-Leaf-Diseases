import os

# Define paths for each test folder
test_folders = [
    r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\5_Test\01_Normal_Test',
    r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\5_Test\02_LeafSpot_Test',
    r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\5_Test\03_MosaicVirus_Test',
    r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\5_Test\04_PowderyMildew_Test'
]

# Initialize counter for continuous numbering
file_counter = 100

# Loop through each folder and rename images
for folder in test_folders:
    for filename in os.listdir(folder):
        # Check if the file is an image
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            # Construct full path for the original file
            original_path = os.path.join(folder, filename)
            
            # New filename with continuous numbering
            new_filename = f"{file_counter}.jpg"
            new_path = os.path.join(folder, new_filename)
            
            # Rename the file
            os.rename(original_path, new_path)
            print(f"Renamed {original_path} to {new_path}")
            
            # Increment counter
            file_counter += 1

print("Renaming completed for all test images.")
