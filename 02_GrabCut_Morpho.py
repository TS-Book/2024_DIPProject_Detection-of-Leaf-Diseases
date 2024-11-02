import cv2
import numpy as np
import os

# กำหนดเส้นทางของไฟล์ภาพสำหรับแต่ละ Class
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

# นามสกุลไฟล์ที่ต้องการตรวจสอบ
extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

# วนลูปประมวลผลแต่ละ Class
for class_name, paths in class_paths.items():
    os.makedirs(paths['output'], exist_ok=True)

    for i in range(1, 401):  # 001-400
        # ใช้ zfill เพื่อให้หมายเลขไฟล์มีสามหลัก
        if class_name == "Normal Leaf":
            base_filename = f"01_Normal_{str(i).zfill(3)}"
        elif class_name == "Leaf Spot":
            base_filename = f"02_Leaf_spot_{str(i).zfill(3)}"
        elif class_name == "Mosaic Virus":
            base_filename = f"03_Mosaic_Virus_{str(i).zfill(3)}"
        elif class_name == "Powdery Mildew":
            base_filename = f"04_Powdery_Mildew_{str(i).zfill(3)}"

        # ตรวจสอบทุกนามสกุล
        loaded = False
        for ext in extensions:
            image_filename = f"{base_filename}{ext}"
            image_path = os.path.join(paths['input'], image_filename)

            # โหลดภาพ
            image = cv2.imread(image_path)
            if image is not None:
                loaded = True
                print(f"Successfully loaded image: {image_path}")

                # ปรับขนาดภาพก่อนการประมวลผล
                image = cv2.resize(image, (256, 256))

                # ปรับปรุงก่อนการประมวลผล
                image = cv2.GaussianBlur(image, (5, 5), 0)  # ใช้ Gaussian Blur เพื่อลด Noise

                # แปลงสีภาพเป็น HSV
                hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

                # ปรับความสว่างและความคอนทราสต์
                image_eq = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

                # การ Thresholding
                _, thresh = cv2.threshold(image_eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # ใช้ Morphological Operations เพื่อทำความสะอาดภาพ
                kernel = np.ones((5, 5), np.uint8)
                morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

                # ใช้ GrabCut
                height, width = image.shape[:2]
                rect = (5, 5, width - 10, height - 10)  # ปรับตำแหน่งตามความเหมาะสม
                
                # เตรียม mask และตัวแปรสำหรับ GrabCut
                mask = np.zeros(image.shape[:2], np.uint8)
                bgd_model = np.zeros((1, 65), np.float64)
                fgd_model = np.zeros((1, 65), np.float64)

                # ใช้ GrabCut
                iterations = 10  # เพิ่มจำนวน iterations
                cv2.grabCut(image, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)

                # แปลง mask
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

                # ใช้ Morphological Operations เพื่อปิดรูที่อยู่ในวัตถุ
                mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)

                # เซฟภาพที่ประมวลผล
                segmented_image = image * mask2[:, :, np.newaxis]
                output_file = os.path.join(paths['output'], os.path.basename(image_path))  # ตั้งชื่อไฟล์ตามต้นฉบับ
                cv2.imwrite(output_file, segmented_image)

                break  # หยุดตรวจสอบนามสกุลเมื่อโหลดไฟล์สำเร็จ
        
        if not loaded:
            print(f"Error loading image for base filename: {base_filename}")

print("Processing completed.")
