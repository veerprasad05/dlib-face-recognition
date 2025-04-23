import cv2
import csv
import face_recognition
import pickle
import numpy as np
from facenet_pytorch import MTCNN

# Load face encodings
with open("trainer/encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_ids = data["ids"]

names = {}
with open("users.csv", 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        id = int(row[0])
        name = row[1]
        names[id] = name
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# Optional: MTCNN to crop faces first (can skip and just use Dlib directly)
mtcnn = MTCNN(keep_all=True, select_largest=False)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')  # or 'cnn'

    encodings = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), encoding in zip(boxes, encodings):
        matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.45)
        distances = face_recognition.face_distance(known_encodings, encoding)

        label = "unknown"
        confidence = 0

        if True in matches:
            best_match_index = np.argmin(distances)
            if matches[best_match_index]:
                user_id = known_ids[best_match_index]
                label = names[user_id] if user_id < len(names) else f"User {user_id}"
                confidence = round((1 - distances[best_match_index]) * 100, 2)

        # Draw results
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"{label}", (left + 5, top - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"{confidence}%", (left + 5, bottom + 20), font, 0.7, (255, 255, 0), 1)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(10) & 0xff == 27:
        break

cam.release()
cv2.destroyAllWindows()
