import time
import cv2
import mediapipe as mp
import pyautogui
import threading

# Import your angle calculation function
from utils.angle import get_angle


def update_gui():
    """Separate thread for updating the GUI."""
    while True:
        if frame_to_show is not None:
            cv2.imshow('Hand Gesture Mouse', frame_to_show)
            cv2.waitKey(1)


def main():
    global frame_to_show  # Global variable to share frame with the GUI thread
    frame_to_show = None

    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()

    prev_mouse_x, prev_mouse_y = 0, 0
    SMOOTHING = 5  # Adjust smoothing factor
    frame_count = 0
    FRAME_SKIP = 2  # Process every nth frame
    prev_frame_time = 0

    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        _, frame = cap.read()
        if frame_count % FRAME_SKIP == 0:  # Skip frames to increase FPS
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (640, 480))  # Resize to reduce processing
            frame_height, frame_width, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            output = hand_detector.process(rgb_frame)
            hands = output.multi_hand_landmarks

            if hands:
                for hand in hands:
                    landmarks = hand.landmark

                    palm_x, palm_y = 0, 0
                    thumb_x, thumb_y = 0, 0
                    index_x, index_y = 0, 0

                    # Extract specific landmarks
                    for id, landmark in enumerate(landmarks):
                        x = int(landmark.x * frame_width)
                        y = int(landmark.y * frame_height)

                        if id == 0:  # Palm
                            palm_x, palm_y = x, y
                            cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)

                        if id == 4:  # Thumb tip
                            thumb_x, thumb_y = x, y
                            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

                        if id == 8:  # Index finger tip
                            index_x, index_y = x, y
                            cv2.circle(frame, (x, y), 10, (255, 0, 255), -1)

                    # Draw lines and calculate angle
                    cv2.line(frame, (palm_x, palm_y), (thumb_x, thumb_y), (0, 0, 255), 2)
                    cv2.line(frame, (palm_x, palm_y), (index_x, index_y), (0, 0, 255), 2)

                    angle = int(get_angle((index_x, index_y), (palm_x, palm_y), (thumb_x, thumb_y)))
                    cv2.putText(frame, f"{angle}", (palm_x, palm_y), font, 1, (100, 255, 0), 1, cv2.LINE_AA)

                    # Smooth mouse movement
                    mouse_x = screen_width / frame_width * palm_x
                    mouse_y = screen_height / frame_height * palm_y
                    if abs(mouse_x - prev_mouse_x) > SMOOTHING or abs(mouse_y - prev_mouse_y) > SMOOTHING:
                        pyautogui.moveTo(mouse_x, mouse_y)
                        prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

                    # Mouse click actions
                    if 10 > angle:
                        pyautogui.rightClick()
                    elif angle > 350:
                        pyautogui.leftClick()

            # Display FPS
            new_frame_time = time.time()
            fps = int(1 / (new_frame_time - prev_frame_time)) if prev_frame_time else 0
            prev_frame_time = new_frame_time
            cv2.putText(frame, f"FPS: {fps}", (7, 37), font, 1, (100, 255, 0), 3, cv2.LINE_AA)

            frame_to_show = frame  # Share the frame with the GUI thread

        frame_count += 1


if __name__ == '__main__':
    frame_to_show = None  # Shared frame variable
    gui_thread = threading.Thread(target=update_gui, daemon=True)
    gui_thread.start()
    main()
