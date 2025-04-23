import cv2
import os
import csv
from facenet_pytorch import MTCNN
import glob

dataset_dir = "dataset"
users_file = "users.csv"

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

# MTCNN detector
mtcnn = MTCNN(select_largest=True, post_process=False)

# Create folders/files if missing
os.makedirs(dataset_dir, exist_ok=True)
for file in glob.glob(os.path.join(dataset_dir, "*.jpg")):
    os.remove(file)
print("[INFO] Cleared existing images from dataset...")

with open(users_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["id", "name"])

# Load existing user mappings
with open(users_file, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    existing_users = list(reader)

# Determine next available ID
used_ids = [int(row[0]) for row in existing_users]
next_id = max(used_ids) + 1 if used_ids else 0

while True:
    name = input("Enter your name: ").strip()
    if not name:
        print("[WARN] Name can't be empty.")
        continue

    user_id = next_id
    next_id += 1

    # Save to CSV
    with open(users_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([user_id, name])

    print(f"\n[INFO] Assigned ID {user_id} to {name}. Capturing 60 face images...")

    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            break

        boxes, probs = mtcnn.detect(frame)
        pad = 10
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                x1, y1 = max(x1 - pad, 0), max(y1 - pad, 0)
                x2, y2 = min(x2 + pad, frame.shape[1]), min(y2 + pad, frame.shape[0])

                face_bgr = frame[y1:y2, x1:x2]
                img_path = f"{dataset_dir}/User.{user_id}.{count + 1}.jpg"
                cv2.imwrite(img_path, face_bgr)
                count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("Capturing", frame)

                if count >= 60:
                    break

        if cv2.waitKey(100) & 0xff == 27 or count >= 60:
            break

    print(f"[INFO] Done capturing for {name} (ID {user_id})")

    # Continue or not
    another = input("Add another user? (y/n): ").lower()
    if another != 'y':
        break

cam.release()
cv2.destroyAllWindows()
print("[INFO] Capture complete.")