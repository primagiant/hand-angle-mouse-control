import cv2
import mediapipe as mp
import pyautogui

from model.hand_gesture_classifier import HandGestureClassifier
from utils.draw import get_landmarks_angle

class_name = ['fist', 'index', 'middle', 'palm', 'two_fingger_close', 'v_gest']
def main():
    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    hand_classifier = HandGestureClassifier()
    screen_width, screen_height = pyautogui.size()

    prev_mouse_x, prev_mouse_y = 0, 0
    SMOOTHING = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hand_detector.process(rgb_frame)
        if results.multi_hand_landmarks:
            landmark_list, coordinates_list = get_landmarks_angle(frame.shape, results.multi_hand_landmarks)
            gesture_id = hand_classifier(landmark_list)

            palm_x = coordinates_list[0][0]
            palm_y = coordinates_list[0][1]
            mouse_x = screen_width / frame_width * palm_x
            mouse_y = screen_height / frame_height * palm_ya

            if abs(mouse_x - prev_mouse_x) > SMOOTHING or abs(mouse_y - prev_mouse_y) > SMOOTHING:
                pyautogui.moveTo(mouse_x, mouse_y)
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

            if class_name[gesture_id] == 'v_gest':
                continue
            elif class_name[gesture_id] == 'index':
                pyautogui.rightClick()
            elif class_name[gesture_id] == 'fist':
                pyautogui.mouseDown()
            elif class_name[gesture_id] == 'middle':
                pyautogui.leftClick()
            elif class_name[gesture_id] == 'palm':
                pyautogui.mouseUp()
            elif class_name[gesture_id] == 'two_fingger_close':
                pyautogui.doubleClick()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
