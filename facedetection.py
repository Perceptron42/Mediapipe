import cv2
import mediapipe as mp
import time

mp_face_detector = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# For webcam input:
cap = cv2.VideoCapture(0)

with mp_face_detector.FaceDetection(min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():

        success, image = cap.read()

        start = time.time() # for calculating fps

        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = face_detection.process(image)

        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        # Convert the image color back so it can be displayed
        # Process the image and find faces
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.detections:
            for id, detection in enumerate(results.detections):
                mp_draw.draw_detection(image, detection)
                # print(id,detection) # id for the face
                print (detection.score[0])
                print (detection.location_data.relative_bounding_box)
                # print(detection.location_data.relative_bounding_box)  # id for the face
                bBox = detection.location_data.relative_bounding_box
                # print(bBox)
                h,w,c = image.shape
                boundBox = int(bBox.xmin *w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
                cv2.putText(image, f'{int(detection.score[0]*100)}%', (boundBox[0], boundBox[1]-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255,2),2)
        end = time.time()
        totalTime = end - start

        fps = 1 / totalTime
        # print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Face Detection', image)

        if cv2.waitKey(5) & 0xFF == 27: # hit excape (ASCII code 27) to break the loop
            break
        # The waitKey(0) function returns -1 when no input is made whatsoever. As soon the event occurs i.e. a Button is pressed it returns a 32-bit integer.
        # The 0xFF in this scenario is representing binary 11111111 a 8 bit binary, since we only require 8 bits to represent a character we AND waitKey(0) to 0xFF. As a result, an integer is obtained below 255.
        # ord(char) returns the ASCII value of the character which would be again maximum 255.
        # Hence by comparing the integer to the ord(char) value, we can check for a key pressed event and break the loop.

cap.release()