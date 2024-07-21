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


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    return number, mode


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/dataset.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


def draw_info(image, mode, number):
    mode_string = ['Normal', 'Pengumpulan Dataset']
    if mode == 0:
        cv2.putText(image, "MODE:" + mode_string[mode], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)
    if mode == 1:
        cv2.putText(image, "MODE:" + mode_string[mode], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                    cv2.LINE_AA)
        if 0 <= number <= 9:
            cv2.putText(image, "NUM:" + str(number), (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1,
                        cv2.LINE_AA)
    return image


def get_landmarks(image, hands):
    landmark_angle = []

    for hand in hands:
        landmarks = hand.landmark
        frame_height, frame_width, _ = image.shape

        palm_coor = (0, 0)

        base_thumb_coor = (0, 0)
        middle_thumb_coor = (0, 0)
        peak_thumb_coor = (0, 0)

        base_index_coor = (0, 0)
        middle_index_coor = (0, 0)
        peak_index_coor = (0, 0)

        base_middle_coor = (0, 0)
        middle_middle_coor = (0, 0)
        peak_middle_coor = (0, 0)

        base_ring_coor = (0, 0)
        middle_ring_coor = (0, 0)
        peak_ring_coor = (0, 0)

        base_little_coor = (0, 0)
        middle_little_coor = (0, 0)
        peak_little_coor = (0, 0)

        # iterasi setiap landmark pada tangan
        for idh, landmark in enumerate(landmarks):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)

            # telapak tangan
            if idh == 0:
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                palm_coor = (x, y)

            # ibu jari
            if idh == 1:  # pangkal ibu jari
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                base_thumb_coor = (x, y)

            if idh == 2:  # tengah ibu jari
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                middle_thumb_coor = (x, y)

            if idh == 4:  # ujung ibu jari
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                peak_thumb_coor = (x, y)

            # telunjuk
            if idh == 5:  # pangkal telunjuk
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                base_index_coor = (x, y)

            if idh == 6:  # tengah telunjuk
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                middle_index_coor = (x, y)

            if idh == 8:  # ujung telunjuk
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                peak_index_coor = (x, y)

            # jari tengah
            if idh == 9:  # pangkal jari tengah
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                base_middle_coor = (x, y)

            if idh == 10:  # tengah jari tengah
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                middle_middle_coor = (x, y)

            if idh == 12:  # ujung jari tengah
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                peak_middle_coor = (x, y)

            # jari manis
            if idh == 13:  # pangkal jari manis
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                base_ring_coor = (x, y)

            if idh == 14:  # tengah jari manis
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                middle_ring_coor = (x, y)

            if idh == 16:  # ujung jari manis
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                peak_ring_coor = (x, y)

            # jari kelingking
            if idh == 17:  # pangkal jari kelingking
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                base_little_coor = (x, y)

            if idh == 18:  # tengah jari kelingking
                cv2.circle(img=image, center=(x, y), radius=5, color=(0, 255, 255))
                middle_little_coor = (x, y)

            if idh == 20:  # ujung jari kelingking
                cv2.circle(img=image, center=(x, y), radius=5, color=(255, 0, 0))
                peak_little_coor = (x, y)

        # Gambar Garis
        draw_line(image, palm_coor, peak_thumb_coor)
        draw_line(image, palm_coor, peak_index_coor)
        draw_line(image, palm_coor, peak_middle_coor)
        draw_line(image, palm_coor, peak_ring_coor)
        draw_line(image, palm_coor, peak_little_coor)

        draw_line(image, middle_thumb_coor, base_thumb_coor, (255, 200, 200))
        draw_line(image, middle_index_coor, base_index_coor, (255, 200, 200))
        draw_line(image, middle_middle_coor, base_middle_coor, (255, 200, 200))
        draw_line(image, middle_ring_coor, base_ring_coor, (255, 200, 200))
        draw_line(image, middle_little_coor, base_little_coor, (255, 200, 200))

        draw_line(image, middle_thumb_coor, peak_thumb_coor, (255, 200, 200))
        draw_line(image, middle_index_coor, peak_index_coor, (255, 200, 200))
        draw_line(image, middle_middle_coor, peak_middle_coor, (255, 200, 200))
        draw_line(image, middle_ring_coor, peak_ring_coor, (255, 200, 200))
        draw_line(image, middle_little_coor, peak_little_coor, (255, 200, 200))

        # hitung sudut ujung ibu jari dan ujung telunjuk
        peak_thumb_peak_index_angle = int(get_angle(peak_thumb_coor, palm_coor, peak_index_coor))
        landmark_angle.append(peak_thumb_peak_index_angle)
        cv2.putText(image, f"1 : {peak_thumb_peak_index_angle}", (frame_width - 120, 30), FONT, 1, (100, 255, 0), 1,
                    cv2.LINE_AA)

        # hitung sudut ujung telunjuk dan ujung jari tengah
        peak_index_peak_middle_angle = int(get_angle(peak_index_coor, palm_coor, peak_middle_coor))
        landmark_angle.append(peak_index_peak_middle_angle)
        cv2.putText(image, f"2 : {peak_index_peak_middle_angle}", (frame_width - 120, 60), FONT, 1, (100, 255, 0), 1,
                    cv2.LINE_4)

        # hitung sudut ujung jari tengah dan ujung jari manis
        peak_middle_peak_ring_angle = int(get_angle(peak_middle_coor, palm_coor, peak_ring_coor))
        landmark_angle.append(peak_middle_peak_ring_angle)
        cv2.putText(image, f"3 : {peak_middle_peak_ring_angle}", (frame_width - 120, 90), FONT, 1, (100, 255, 0), 1,
                    cv2.LINE_4)

        # hitung sudut ujung jari manis dan ujung jari kelingking
        peak_ring_peak_little_angle = int(get_angle(peak_ring_coor, palm_coor, peak_little_coor))
        landmark_angle.append(peak_ring_peak_little_angle)
        cv2.putText(image, f"4 : {peak_ring_peak_little_angle}", (frame_width - 120, 120), FONT, 1, (100, 255, 0), 1,
                    cv2.LINE_4)

        # Ruas Ruas Jari
        # Ibu Jari
        thumb_finger_join = int(get_angle(base_thumb_coor, middle_thumb_coor, peak_thumb_coor))
        landmark_angle.append(thumb_finger_join)
        cv2.putText(image, f"5 : {thumb_finger_join}", (frame_width - 120, 150), FONT, 1, (100, 255, 0), 1,
                    cv2.LINE_4)

        # Telunjuk
        index_finger_join = int(get_angle(base_index_coor, middle_index_coor, peak_index_coor))
        landmark_angle.append(index_finger_join)
        cv2.putText(image, f"6 : {index_finger_join}", (frame_width - 120, 180), FONT, 1, (100, 255, 0), 1,
                    cv2.LINE_4)


        # Tengah
        middle_finger_join = int(get_angle(base_middle_coor, middle_middle_coor, peak_middle_coor))
        landmark_angle.append(middle_finger_join)
        cv2.putText(image, f"7 : {middle_finger_join}", (frame_width - 120, 210), FONT, 1, (100, 255, 0), 1,
                    cv2.LINE_4)

        # Manis
        ring_finger_join = int(get_angle(base_ring_coor, middle_ring_coor, peak_ring_coor))
        landmark_angle.append(ring_finger_join)
        cv2.putText(image, f"8 : {ring_finger_join}", (frame_width - 120, 240), FONT, 1, (100, 255, 0), 1,
                    cv2.LINE_4)

        # Kelingking
        little_finger_join = int(get_angle(base_little_coor, middle_little_coor, peak_little_coor))
        landmark_angle.append(little_finger_join)
        cv2.putText(image, f"9 : {little_finger_join}", (frame_width - 120, 270), FONT, 1, (100, 255, 0), 1,
                    cv2.LINE_4)

    return landmark_angle


if __name__ == '__main__':
    main()
