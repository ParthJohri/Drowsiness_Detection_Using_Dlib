import cv2
import dlib
import playsound
import numpy as np
import time

video_path = './testing.mov' 

modelPath = "models/shape_predictor_70_face_landmarks.dat"
sound_path = "alarm.wav"

FACE_DOWNSAMPLE_RATIO = 1.5
RESIZE_HEIGHT = 460
thresh = 0.27
blinkTime = 0.15  
drowsyTime = 1.5  
falseBlinkLimit = 10
drowsyLimit = 100

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

leftEyeIndex = [36, 37, 38, 39, 40, 41]
rightEyeIndex = [42, 43, 44, 45, 46, 47]

blinkCount = 0
drowsy = 0
state = 0
ALARM_ON = False

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = './output_video.avi'
output_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (640, 480))

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def checkEyeStatus(landmarks):
    mask = np.zeros(frame.shape[:2], dtype=np.float32)

    hullLeftEye = [landmarks[i] for i in leftEyeIndex]
    cv2.fillConvexPoly(mask, np.int32(hullLeftEye), 255)

    hullRightEye = [landmarks[i] for i in rightEyeIndex]
    cv2.fillConvexPoly(mask, np.int32(hullRightEye), 255)

    leftEAR = eye_aspect_ratio(hullLeftEye)
    rightEAR = eye_aspect_ratio(hullRightEye)
    ear = (leftEAR + rightEAR) / 2.0

    eyeStatus = 1  # Open
    if ear < thresh:
        eyeStatus = 0  # Closed

    return eyeStatus

def checkBlinkStatus(eyeStatus):
    global state, blinkCount, drowsy
    if state >= 0 and state <= falseBlinkLimit:
        if eyeStatus:
            state = 0
        else:
            state += 1
    elif state >= falseBlinkLimit and state < drowsyLimit:
        if eyeStatus:
            blinkCount += 1
            state = 0
        else:
            state += 1
    else:
        if eyeStatus:
            state = 0
            drowsy = 1
            blinkCount += 1
        else:
            drowsy = 1

def getLandmarks(im):
    imSmall = cv2.resize(im, None, fx=1.0/FACE_DOWNSAMPLE_RATIO, fy=1.0/FACE_DOWNSAMPLE_RATIO, interpolation=cv2.INTER_LINEAR)
    rects = detector(imSmall, 0)
    if len(rects) == 0:
        return 0

    newRect = dlib.rectangle(int(rects[0].left() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].top() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].right() * FACE_DOWNSAMPLE_RATIO),
                             int(rects[0].bottom() * FACE_DOWNSAMPLE_RATIO))

    points = []
    [points.append((p.x, p.y)) for p in predictor(im, newRect).parts()]
    return points

def main():
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        print("Error: Unable to open video file.")
        return

    while True:
        try:
            ret, frame = capture.read()
            if not ret:
                print("End of video.")
                break

            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            frame = cv2.resize(frame, None, fx=1/IMAGE_RESIZE, fy=1/IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)

            landmarks = getLandmarks(frame)
            if landmarks == 0:
                continue

            eyeStatus = checkEyeStatus(landmarks)
            checkBlinkStatus(eyeStatus)

            # Display blink count or drowsiness alert if needed

            cv2.imshow("Frame", frame)
            output_video.write(frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        except Exception as e:
            print(e)

    capture.release()
    output_video.release()
    cv2.destroyAllWindows()

