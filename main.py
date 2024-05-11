import math

import cv2
import mediapipe as mp
import pyautogui
import time

cap = cv2.VideoCapture(0)
hand_detector = mp.solutions.hands.Hands(
    max_num_hands=1
)
drawing_utils = mp.solutions.drawing_utils
screen_width, screen_height = pyautogui.size()

prev_frame_time = 0
new_frame_time = 0

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame_height, frame_width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hand_detector.process(rgb_frame)
    hands = output.multi_hand_landmarks

    if hands:
        for hand in hands:
            drawing_utils.draw_landmarks(frame, hand)
            landmarks = hand.landmark

            dis_palm_to_thumb = 0
            dis_palm_to_index = 0

            palm_x, palm_y = 0, 0
            thumb_x, thumb_y = 0, 0
            index_x, index_y = 0, 0

            # iterasi setiap landmark pada tangan
            for id, landmark in enumerate(landmarks):
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)

                if id == 0:  # telapak tangan
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 255))
                    palm_x, palm_y = x, y
                    # mouse_x = screen_width/frame_width * x
                    # mouse_y = screen_height/frame_height * y
                    # pyautogui.moveTo(mouse_x, mouse_y)

                if id == 4:  # ibu jari
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 0))
                    thumb_x, thumb_y = x, y

                if id == 8:  # telunjuk
                    cv2.circle(img=frame, center=(x, y), radius=10, color=(255, 0, 255))
                    index_x, index_y = x, y

            # hitung jarak antara telapak tangan dengan ibu jari
            dis_palm_to_thumb = math.sqrt((thumb_x - palm_x) ** 2 + (thumb_y - palm_y) ** 2)
            cv2.line(img=frame, pt1=(palm_x, palm_y), pt2=(thumb_x, thumb_y), color=(0, 0, 255), thickness=2)

            # hitung jarak antara telapak tangan dengan telunjuk
            dis_palm_to_index = math.sqrt((index_x - palm_x) ** 2 + (index_y - palm_y) ** 2)
            cv2.line(img=frame, pt1=(palm_x, palm_y), pt2=(index_x, index_y), color=(0, 0, 255), thickness=2)

            # hitung sudut
            a = (dis_palm_to_thumb * dis_palm_to_index)
            b = (abs(dis_palm_to_thumb) * abs(dis_palm_to_index))
            angle = math.acos(a / b)
            print(a, b, angle)
            # cv2.putText(frame, angle, (palm_x, palm_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

    # FPS SHOW
    font = cv2.FONT_HERSHEY_SIMPLEX
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    cv2.putText(frame, f"FPS : {fps}", (7, 37), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Hand Gesture Mouse', frame)
    cv2.waitKey(1)
