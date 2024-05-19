import copy
import csv

import mediapipe as mp

from utils.angle import *
from utils.draw import *

FONT = cv2.FONT_HERSHEY_SIMPLEX


def main():
    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(
        max_num_hands=1
    )

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
            logging_csv(number, mode, landmark_list)

        debug_image = draw_info(debug_image, mode, number)
        cv2.imshow('Hand Gesture Mouse', debug_image)

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
        thumb_coor = (0, 0)
        index_coor = (0, 0)
        middle_coor = (0, 0)
        ring_coor = (0, 0)
        little_coor = (0, 0)

        # iterasi setiap landmark pada tangan
        for idh, landmark in enumerate(landmarks):
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)

            if idh == 0:  # telapak tangan
                cv2.circle(img=image, center=(x, y), radius=10, color=(0, 255, 255))
                palm_coor = (x, y)

            if idh == 4:  # ibu jari
                cv2.circle(img=image, center=(x, y), radius=10, color=(255, 0, 255))
                thumb_coor = (x, y)

            if idh == 8:  # telunjuk
                cv2.circle(img=image, center=(x, y), radius=10, color=(255, 0, 255))
                index_coor = (x, y)

            if idh == 12:  # tengah
                cv2.circle(img=image, center=(x, y), radius=10, color=(255, 0, 255))
                middle_coor = (x, y)

            if idh == 16:  # manis
                cv2.circle(img=image, center=(x, y), radius=10, color=(255, 0, 255))
                ring_coor = (x, y)

            if idh == 20:  # kelingking
                cv2.circle(img=image, center=(x, y), radius=10, color=(255, 0, 255))
                little_coor = (x, y)

        ### telunjuk dan ibu jari
        # gambar garis yang merepresentasikan sudut
        draw_line(image, palm_coor, thumb_coor)
        draw_line(image, palm_coor, index_coor)
        # hitung sudut telunjuk dan ibu jari
        thumb_index_angle = int(get_angle(thumb_coor, palm_coor, index_coor))
        landmark_angle.append(thumb_index_angle)
        cv2.putText(image, f"1 : {thumb_index_angle}", (frame_width - 120, 30), FONT, 1, (100, 255, 0), 1, cv2.LINE_AA)

        ### telunjuk dan tengah
        # gambar garis yang merepresentasikan sudut
        draw_line(image, palm_coor, index_coor)
        draw_line(image, palm_coor, middle_coor)
        # hitung sudut telunjuk dan ibu jari
        index_middle_angle = int(get_angle(index_coor, palm_coor, middle_coor))
        landmark_angle.append(index_middle_angle)
        cv2.putText(image, f"2 : {index_middle_angle}", (frame_width - 120, 60), FONT, 1, (100, 255, 0), 1, cv2.LINE_4)

        ### tengah dan manis
        # gambar garis yang merepresentasikan sudut
        draw_line(image, palm_coor, middle_coor)
        draw_line(image, palm_coor, ring_coor)
        # hitung sudut tengah dan manis
        middle_ring_angle = int(get_angle(middle_coor, palm_coor, ring_coor))
        landmark_angle.append(middle_ring_angle)
        cv2.putText(image, f"3 : {middle_ring_angle}", (frame_width - 120, 90), FONT, 1, (100, 255, 0), 1, cv2.LINE_4)

        ### manis dan kelingking
        # gambar garis yang merepresentasikan sudut
        draw_line(image, palm_coor, ring_coor)
        draw_line(image, palm_coor, little_coor)
        # hitung sudut manis dan kelingking
        ring_little_angle = int(get_angle(ring_coor, palm_coor, little_coor))
        landmark_angle.append(ring_little_angle)
        cv2.putText(image, f"4 : {ring_little_angle}", (frame_width - 120, 120), FONT, 1, (100, 255, 0), 1, cv2.LINE_4)

    return landmark_angle


if __name__ == '__main__':
    main()
