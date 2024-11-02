import os

# กำหนดโฟลเดอร์ที่เก็บรูปภาพ
folders = [
    r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_Raw_Images\01_Normal",
    r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_Raw_Images\02_Leaf spot",
    r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_Raw_Images\03_Mosaic Virus",
    r"D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\1_Raw_Images\04_Powdery Mildew",
]

# ตั้งค่าชื่อใหม่ที่ต้องการ
new_name_prefixes = [
    "01_Normal",
    "02_Leaf_spot",
    "03_Mosaic_Virus",
    "04_Powdery_Mildew",
    "05_Test"
]

# เริ่มต้น index สำหรับแต่ละโฟลเดอร์
for folder_path, new_name_prefix in zip(folders, new_name_prefixes):
    start_index = 1  # เริ่มต้นนับจาก 1

    # วนลูปเพื่อเปลี่ยนชื่อไฟล์ทีละไฟล์
    for filename in os.listdir(folder_path):
        # ตรวจสอบว่าไฟล์เป็นไฟล์ภาพ
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG')):
            # กำหนดชื่อใหม่
            new_filename = f"{new_name_prefix}_{str(start_index).zfill(3)}.jpg"  # เลขจะเป็น 3 หลัก เช่น 001, 002, ...
            
            # สร้าง path สำหรับไฟล์เดิมและไฟล์ใหม่
            old_file_path = os.path.join(folder_path, filename)
            new_file_path = os.path.join(folder_path, new_filename)
            
            # เปลี่ยนชื่อไฟล์
            os.rename(old_file_path, new_file_path)
            
            # เพิ่ม index เพื่อไม่ให้ชื่อซ้ำกัน
            start_index += 1

    print(f"เปลี่ยนชื่อไฟล์ในโฟลเดอร์ '{folder_path}' เสร็จสิ้น")

print("เปลี่ยนชื่อไฟล์เสร็จสิ้นทั้งหมด")
