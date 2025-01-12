import copy
import csv

import cv2
import mediapipe as mp

from utils.angle import *
from utils.draw import *
from model.hand_gesture_classifier import HandGestureClassifier

class_name = ['fist', 'index', 'middle', 'palm', 'two_fingger_close', 'v_gest']

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(
        max_num_hands=1
    )

    hand_classifier = HandGestureClassifier()

    mode = 0

    while True:

        # Process Key (ESC: end)
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hand_detector.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            landmark_list = get_landmarks(debug_image, results.multi_hand_landmarks)
            hand_sign_id = hand_classifier(landmark_list)
            cv2.putText(debug_image, "CLASSIFY:" + class_name[hand_sign_id], (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                        cv2.LINE_AA)
            logging_csv(number, mode, landmark_list)

        debug_image = draw_info(debug_image, mode, number)
        cv2.imshow('Hand Gesture Mouse', debug_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
