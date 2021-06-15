#!/usr/bin/env python3
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Or If the input is the camera, pass 0 instead of the video file
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5, max_num_hands=1) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 2), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            for item in results.multi_handedness:
                print(item)
                cv2.putText(image, "LABEL: " + str(item.classification[0].label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "SCORE: " + str(int(item.classification[0].score*100)), (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 2, cv2.LINE_AA)
            for id, hand_landmarks in enumerate(results.multi_hand_landmarks[0].landmark):
                if id == 8:
                    print(id, hand_landmarks)
                    cv2.circle(image, (int(hand_landmarks.x*640), int(hand_landmarks.y*480)), 10,
                               (152, 251, 152), 2)

                # for point in mp_hands.HandLandmark:
                #     print(point)
                #     print(hand_landmarks.landmark[point])

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
