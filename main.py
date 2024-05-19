import time

import cv2
import mediapipe as mp
import pyautogui

from utils.angle import *


def main():
    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(
        max_num_hands=1
    )
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()

    prev_frame_time = 0
    new_frame_time = 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            for hand in hands:
                # drawing_utils.draw_landmarks(frame, hand)
                landmarks = hand.landmark

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

                    if id == 4:  # ibu jari
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(0, 255, 0))
                        thumb_x, thumb_y = x, y

                    if id == 8:  # telunjuk
                        cv2.circle(img=frame, center=(x, y), radius=10, color=(255, 0, 255))
                        index_x, index_y = x, y

                # gambar garis yang merepresentasikan sudut
                cv2.line(img=frame, pt1=(palm_x, palm_y), pt2=(thumb_x, thumb_y), color=(0, 0, 255), thickness=2)
                cv2.line(img=frame, pt1=(palm_x, palm_y), pt2=(index_x, index_y), color=(0, 0, 255), thickness=2)

                # hitung sudut
                angle = int(get_angle((index_x, index_y), (palm_x, palm_y), (thumb_x, thumb_y)))
                cv2.putText(frame, f"{angle}", (palm_x, palm_y), font, 1, (100, 255, 0), 1, cv2.LINE_AA)

                # gerakan mouse
                mouse_x = screen_width / frame_width * palm_x
                mouse_y = screen_height / frame_height * palm_y
                pyautogui.moveTo(mouse_x, mouse_y)

                # Right Click
                if 10 > angle:
                    pyautogui.rightClick()

                # Left Click
                if angle > 350:
                    pyautogui.leftClick()
                else:
                    pyautogui.mouseUp()

        # FPS SHOW
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv2.putText(frame, f"FPS : {fps}", (7, 37), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

        cv2.imshow('Hand Gesture Mouse', frame)
        cv2.waitKey(1)


if __name__ == '__main__':
    main()
