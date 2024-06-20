from scipy.spatial import distance
import cv2
import dlib
from imutils import face_utils
import imutils
import os
import time
shape_predictor_path = "./models/shape_predictor_68_face_landmarks_onlyeyes_4depth_15cascadedepth.dat"
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor_path)
start_time = time.time()

image_dir = "./testing"
output_dir = "./example"
EAR_THRESHOLD_DROWSY = 0.2 

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)
    return ear
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".tif") or filename.endswith("jpeg"):  # Add other extensions if needed
        image_path = os.path.join(image_dir, filename)
        print('image_path: ', image_path)
        image = cv2.imread(image_path)

        image = imutils.resize(image, width=400)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:

            (x, y, w, h) = face_utils.rect_to_bb(rect)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            print('shape: ', shape)


            left_eye = shape[0:6]
            right_eye = shape[6:12]

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)

            avg_ear = (left_ear + right_ear) / 2

            status = "awake" if avg_ear > EAR_THRESHOLD_DROWSY else "drowsy"
            color =  (0, 0, 255) if status == "drowsy" else (0, 255, 0)

            for (sX, sY) in shape:
                cv2.circle(image, (sX, sY), 1, (0, 0, 255), -1)
            
            cv2.putText(image, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, image)
        print(f"[INFO] Processed and saved {filename}")
        
end_time = time.time()

inference_time = end_time - start_time
print(f'Inference Time: {inference_time} seconds')

with open('inference_time.txt', 'w') as file:
    file.write(f'Inference Time: {inference_time} seconds\n')
print("[INFO] All images have been processed and saved.")
