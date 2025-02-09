import copy
import cv2
import mediapipe as mp
import pyautogui
import time

from utils.angle import *
from utils.draw import *
from model.hand_gesture_random_classifier import HandGestureRandomClassifier

class_name = ['fist', 'index', 'middle', 'palm', 'two_fingger_close', 'v_gest', 'random']

FONT = cv2.FONT_HERSHEY_SIMPLEX

# Define boundary box for mouse control (x1, y1, x2, y2)
BOUNDARY = (100, 100, 500, 400)  # Left-Top to Right-Bottom
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# Smoothing parameters
alpha = 0.2  # Smoothing factor (higher = more responsive, lower = smoother)
prev_screen_x, prev_screen_y = None, None

# Click debounce timers
last_click_time = {"left": 0, "right": 0, "double": 0}
click_delay = 0.5  # Minimum delay (seconds) between consecutive clicks

# Gesture stability
gesture_memory = {}
gesture_threshold = 5  # Frames required to confirm a gesture

def is_inside_boundary(x, y):
    """Check if the hand center is inside the defined boundary"""
    x1, y1, x2, y2 = BOUNDARY
    return x1 <= x <= x2 and y1 <= y <= y2


def map_to_screen(x, y):
    """Map hand position inside boundary to screen coordinates"""
    x1, y1, x2, y2 = BOUNDARY
    screen_x = ((x - x1) / (x2 - x1)) * SCREEN_WIDTH
    screen_y = ((y - y1) / (y2 - y1)) * SCREEN_HEIGHT
    return int(screen_x), int(screen_y)


def smooth_position(new_x, new_y):
    global prev_screen_x, prev_screen_y
    if prev_screen_x is None or prev_screen_y is None:
        prev_screen_x, prev_screen_y = new_x, new_y

    smooth_x = int(alpha * new_x + (1 - alpha) * prev_screen_x)
    smooth_y = int(alpha * new_y + (1 - alpha) * prev_screen_y)

    prev_screen_x, prev_screen_y = smooth_x, smooth_y
    return smooth_x, smooth_y


def check_and_click(gesture, click_type):
    global last_click_time, gesture_memory

    if gesture not in gesture_memory:
        gesture_memory[gesture] = 0

    gesture_memory[gesture] += 1
    if gesture_memory[gesture] >= gesture_threshold:
        current_time = time.time()
        if current_time - last_click_time[click_type] > click_delay:
            if click_type == "left":
                pyautogui.leftClick()
            elif click_type == "right":
                pyautogui.rightClick()
            elif click_type == "double":
                pyautogui.doubleClick()
            last_click_time[click_type] = current_time
        gesture_memory[gesture] = 0  # Reset memory after execution


def main():
    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(max_num_hands=1)
    hand_classifier = HandGestureRandomClassifier()

    while True:
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break

        ret, image = cap.read()
        if not ret:
            break

        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hand_detector.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks:
            landmark_list, coordinates_list = get_landmarks_angle(debug_image.shape, results.multi_hand_landmarks)
            hand_sign_id = hand_classifier(landmark_list)

            palm_x, palm_y = coordinates_list[0][0], coordinates_list[0][1]

            if is_inside_boundary(palm_x, palm_y):
                screen_x, screen_y = map_to_screen(palm_x, palm_y)
                smooth_x, smooth_y = smooth_position(screen_x, screen_y)
                pyautogui.moveTo(smooth_x, smooth_y, duration=0.05)

                if class_name[hand_sign_id] == 'index':
                    check_and_click("index", "right")
                elif class_name[hand_sign_id] == 'fist':
                    pyautogui.mouseDown()
                elif class_name[hand_sign_id] == 'middle':
                    check_and_click("middle", "left")
                elif class_name[hand_sign_id] == 'palm':
                    pyautogui.mouseUp()
                elif class_name[hand_sign_id] == 'two_fingger_close':
                    check_and_click("two_fingger_close", "double")

            cv2.putText(debug_image, "CLASSIFY: " + class_name[hand_sign_id], (10, 30),
                        FONT, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw boundary box
        x1, y1, x2, y2 = BOUNDARY
        cv2.rectangle(debug_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.imshow('Hand Gesture Mouse', debug_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
