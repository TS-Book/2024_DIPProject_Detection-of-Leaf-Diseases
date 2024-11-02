import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Load HOG and color histogram features for each class
normal_features = np.load(r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\3_Hog\01_normal\HOG_ColorHist_features_Normal.npy')
leaf_spot_features = np.load(r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\3_Hog\02_Leaf_Spot\HOG_ColorHist_features_Leaf_Spot.npy')
mosaic_virus_features = np.load(r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\3_Hog\03_Mosaic_Virus\HOG_ColorHist_features_Mosaic_Virus.npy')
powdery_mildew_features = np.load(r'D:\University\3\3_1\DIP\Mini project\Plant_Disease_Detection\Data\3_Hog\04_Powdery_Mildew\HOG_ColorHist_features_Powdery_Mildew.npy')

# Create labels for each class
labels = {
    "Normal": 0,
    "Leaf Spot": 1,
    "Mosaic Virus": 2,
    "Powdery Mildew": 3
}
normal_labels = np.full(normal_features.shape[0], labels["Normal"])
leaf_spot_labels = np.full(leaf_spot_features.shape[0], labels["Leaf Spot"])
mosaic_virus_labels = np.full(mosaic_virus_features.shape[0], labels["Mosaic Virus"])
powdery_mildew_labels = np.full(powdery_mildew_features.shape[0], labels["Powdery Mildew"])

# Combine features and labels
X = np.concatenate((normal_features, leaf_spot_features, mosaic_virus_features, powdery_mildew_features))
y = np.concatenate((normal_labels, leaf_spot_labels, mosaic_virus_labels, powdery_mildew_labels))

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the SVM model
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluate the model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the SVM model and scaler
model_save_path = r'D:\University\3\3_1\DIP\Mini project\TEST_Plant_Disease_Detection\Data\4_SVM\svm_model.joblib'
scaler_save_path = r'D:\University\3\3_1\DIP\Mini project\TEST_Plant_Disease_Detection\Data\4_SVM\scaler.joblib'
dump(svm, model_save_path)
dump(scaler, scaler_save_path)
print(f"Model saved to: {model_save_path}")
print(f"Scaler saved to: {scaler_save_path}")
