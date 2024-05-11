import cv2
import mediapipe as mp

capture = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(
    max_num_hands=2
)
drawing_utils = mp.solutions.drawing_utils

while True:
    _, frame = capture.read()
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            for id, landmark in enumerate(landmarks):
                x = landmark.x
                y = landmark.y

                print(x, y)

    cv2.imshow('Dataset Maker', frame)
    cv2.waitKey(1)
