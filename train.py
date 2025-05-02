import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

DATASET_DIR = "dataset"
IMG_SIZE = (64, 64)

def extract_rust_mask_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_rust = np.array([5, 50, 50])
    upper_rust = np.array([20, 255, 255])
    mask = cv2.inRange(hsv, lower_rust, upper_rust)
    resized = cv2.resize(mask, IMG_SIZE).flatten()
    return resized

def load_data():
    X, y = [], []
    for label, folder in enumerate(["clean", "rust"]):
        path = os.path.join(DATASET_DIR, folder)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            img = cv2.imread(file_path)
            if img is None:
                continue
            feature = extract_rust_mask_features(img)
            X.append(feature)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data()
print(f"Loaded {len(X)} samples")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

print("Classification Report:")
print(classification_report(y_test, clf.predict(X_test)))

joblib.dump(clf, "model.pkl")
print("Model saved.")
