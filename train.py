import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Path dataset
DATASET_DIR = "dataset"
IMG_SIZE = (64, 64)

def load_data():
    X = []
    y = []
    for label, folder in enumerate(["clean", "rust"]):  # 0: clean, 1: rust
        path = os.path.join(DATASET_DIR, folder)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            img = cv2.imread(file_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)
            edges = cv2.Canny(blur, 50, 150)  # Feature extraction
            resized = cv2.resize(edges, IMG_SIZE).flatten()  # Vectorize
            X.append(resized)
            y.append(label)
    return np.array(X), np.array(y)

# Load data
X, y = load_data()
print(f"Dataset loaded: {X.shape[0]} samples")

# Split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, "model.pkl")
print("Model saved as model.pkl")
