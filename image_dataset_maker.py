import csv
import os

import mediapipe as mp

from utils.draw import *

def main():
    hand_detector = mp.solutions.hands.Hands(
        max_num_hands=1
    )
    d_foldername = './dataset'
    for root, dirs, files in os.walk(d_foldername):
        for file in files:
            relative_path = os.path.relpath(os.path.join(root, file), d_foldername)
            pose = os.path.basename(root)
            image = cv2.imread(os.path.join(d_foldername, relative_path))
            image = cv2.flip(image, 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hand_detector.process(image)
            if results.multi_hand_landmarks is not None:
                landmark_list = get_landmarks(image, results.multi_hand_landmarks)
                csv_path = 'model/dataset.csv'
                with open(csv_path, 'a', newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([pose, *landmark_list])
        print("A block of files has been transformed into the dataset")
    print("Completed")



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
                palm_coor = (x, y)

            # ibu jari
            if idh == 1:  # pangkal ibu jari
                base_thumb_coor = (x, y)

            if idh == 2:  # tengah ibu jari
                middle_thumb_coor = (x, y)

            if idh == 4:  # ujung ibu jari
                peak_thumb_coor = (x, y)

            # telunjuk
            if idh == 5:  # pangkal telunjuk
                base_index_coor = (x, y)

            if idh == 6:  # tengah telunjuk
                middle_index_coor = (x, y)

            if idh == 8:  # ujung telunjuk
                peak_index_coor = (x, y)

            # jari tengah
            if idh == 9:  # pangkal jari tengah
                base_middle_coor = (x, y)

            if idh == 10:  # tengah jari tengah
                middle_middle_coor = (x, y)

            if idh == 12:  # ujung jari tengah
                peak_middle_coor = (x, y)

            # jari manis
            if idh == 13:  # pangkal jari manis
                base_ring_coor = (x, y)

            if idh == 14:  # tengah jari manis
                middle_ring_coor = (x, y)

            if idh == 16:  # ujung jari manis
                peak_ring_coor = (x, y)

            # jari kelingking
            if idh == 17:  # pangkal jari kelingking
                base_little_coor = (x, y)

            if idh == 18:  # tengah jari kelingking
                middle_little_coor = (x, y)

            if idh == 20:  # ujung jari kelingking
                peak_little_coor = (x, y)

        # hitung sudut ujung ibu jari dan ujung telunjuk
        peak_thumb_peak_index_angle = int(get_angle(peak_thumb_coor, palm_coor, peak_index_coor))
        landmark_angle.append(peak_thumb_peak_index_angle)

        # hitung sudut ujung telunjuk dan ujung jari tengah
        peak_index_peak_middle_angle = int(get_angle(peak_index_coor, palm_coor, peak_middle_coor))
        landmark_angle.append(peak_index_peak_middle_angle)

        # hitung sudut ujung jari tengah dan ujung jari manis
        peak_middle_peak_ring_angle = int(get_angle(peak_middle_coor, palm_coor, peak_ring_coor))
        landmark_angle.append(peak_middle_peak_ring_angle)

        # hitung sudut ujung jari manis dan ujung jari kelingking
        peak_ring_peak_little_angle = int(get_angle(peak_ring_coor, palm_coor, peak_little_coor))
        landmark_angle.append(peak_ring_peak_little_angle)

        # Ruas Ruas Jari
        # Ibu Jari
        thumb_finger_join = int(get_angle(base_thumb_coor, middle_thumb_coor, peak_thumb_coor))
        landmark_angle.append(thumb_finger_join)

        # Telunjuk
        index_finger_join = int(get_angle(base_index_coor, middle_index_coor, peak_index_coor))
        landmark_angle.append(index_finger_join)

        # Tengah
        middle_finger_join = int(get_angle(base_middle_coor, middle_middle_coor, peak_middle_coor))
        landmark_angle.append(middle_finger_join)

        # Manis
        ring_finger_join = int(get_angle(base_ring_coor, middle_ring_coor, peak_ring_coor))
        landmark_angle.append(ring_finger_join)

        # Kelingking
        little_finger_join = int(get_angle(base_little_coor, middle_little_coor, peak_little_coor))
        landmark_angle.append(little_finger_join)

    return landmark_angle


if __name__ == '__main__':
    main()
