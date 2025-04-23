import os
import cv2
import face_recognition
import numpy as np
import pickle

dataset_path = 'dataset'
encoding_path = 'trainer/encodings.pkl'
os.makedirs('trainer', exist_ok=True)

known_encodings = []
known_ids = []

# Loop through dataset and encode
for filename in os.listdir(dataset_path):
    if not filename.endswith(".jpg"):
        continue

    path = os.path.join(dataset_path, filename)
    image = cv2.imread(path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(rgb, model='hog')  # or 'cnn' if GPU
    encodings = face_recognition.face_encodings(rgb, boxes)

    if len(encodings) > 0:
        encoding = encodings[0]
        user_id = int(filename.split('.')[1])
        known_encodings.append(encoding)
        known_ids.append(user_id)

print(f"[INFO] Encoded {len(known_encodings)} face images.")

# Save encodings to file
data = {"encodings": known_encodings, "ids": known_ids}
with open(encoding_path, "wb") as f:
    pickle.dump(data, f)

print("[INFO] Saved face encodings to trainer/encodings.pkl")
