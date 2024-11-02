import os
import cv2
import numpy as np
import random

# กำหนดโฟลเดอร์ที่มีรูปภาพต้นฉบับและโฟลเดอร์สำหรับเก็บรูปที่ผ่านการ Augmentation
folders = [
    (r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_Raw_Images\01_Normal", r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_1_Augmentation\01_Normal"),
    (r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_Raw_Images\02_Leaf spot", r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_1_Augmentation\02_Leaf_spot"),
    (r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_Raw_Images\03_Mosaic Virus", r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_1_Augmentation\03_Mosaic_Virus"),
    (r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_Raw_Images\04_Powdery Mildew", r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_1_Augmentation\04_Powdery_Mildew")
]

# ฟังก์ชันสำหรับการสุ่ม Augmentation
def random_augment(image):
    # สุ่มเลือกการ Augmentation
    if random.choice([True, False]):
        image = cv2.flip(image, 0)  # Flip แนวตั้ง

    if random.choice([True, False]):
        image = cv2.flip(image, 1)  # Flip แนวนอน

    if random.choice([True, False]):
        angle = random.choice([90, 180, 270])  # หมุนภาพด้วยมุมสุ่ม
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    if random.choice([True, False]):
        alpha = random.uniform(0.8, 1.0)  # ลดช่วง alpha ให้เข้มงวดขึ้น
        beta = random.randint(-10, 0)      # ลดช่วง beta ให้ลดความสว่าง
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    if random.choice([True, False]):
        kernel_size = random.choice([3, 5])  # ปรับ Blur ด้วย Gaussian Blur
        image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    if random.choice([True, False]):
        alpha_contrast = random.uniform(1.0, 2.0)  # ปรับ Contrast แบบสุ่ม
        image = cv2.convertScaleAbs(image, alpha=alpha_contrast, beta=0)

    return image

# วนลูปเพื่อประมวลผลแต่ละโฟลเดอร์
for input_folder, output_folder in folders:
    os.makedirs(output_folder, exist_ok=True)

    # คัดลอกรูปต้นฉบับทั้งหมดไปยัง output_folder
    existing_images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))]
    
    # คัดลอกรูปภาพต้นฉบับไปยัง output_folder
    for i, filename in enumerate(existing_images):
        image = cv2.imread(os.path.join(input_folder, filename))
        new_filename = f"{os.path.basename(output_folder)}_{str(i + 1).zfill(3)}.jpg"  # ตั้งชื่อใหม่ให้เรียงตามลำดับ
        cv2.imwrite(os.path.join(output_folder, new_filename), image)

    # ทำ Augmentation จนกว่าจะได้รูปครบ 400 รูปใน output_folder
    current_image_count = len(existing_images)
    while current_image_count < 1000:
        for filename in existing_images:
            image = cv2.imread(os.path.join(input_folder, filename))
            augmented_image = random_augment(image)

            if current_image_count >= 1000:
                break

            current_image_count += 1
            new_filename = f"{os.path.basename(output_folder)}_{str(current_image_count).zfill(3)}.jpg"
            cv2.imwrite(os.path.join(output_folder, new_filename), augmented_image)

    print(f"Data Augmentation เสร็จสิ้นสำหรับ '{os.path.basename(input_folder)}' มีรูปภาพทั้งหมด 1000 รูปในโฟลเดอร์:", output_folder)

print("Data Augmentation เสร็จสิ้นทั้งหมด")
