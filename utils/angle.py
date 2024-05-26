import numpy as np


def get_angle(a, b, c):
    m1 = gradient(b, a)
    m2 = gradient(b, c)
    if m1 == np.inf or m2 == np.inf:
        if m1 == m2:
            return 0  # Same direction
        else:
            return np.pi / 2  # Perpendicular
    try:
        buffer = (m1 - m2) / (1 + (m1 * m2))
    except ZeroDivisionError:
        buffer = 0
    theta = np.arctan(buffer)
    theta = np.abs(theta)
    theta = np.degrees(theta)

    return theta


def gradient(coor1, coor2):
    x1, y1 = coor1
    x2, y2 = coor2
    try:
        m = (y2 - y1) / (x2 - x1)
        if x1 == x2:
            m = np.inf
    except ZeroDivisionError:
        m = np.inf

    return m
