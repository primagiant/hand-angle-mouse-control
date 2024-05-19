import cv2


def draw_line(frame, pt1, pt2, color=(0, 0, 255)):
    cv2.line(frame, pt1, pt2, color, 1)
