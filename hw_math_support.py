import numpy as np
import math


def line_eq(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    if x1 == x2:
        raise ValueError("points must be different")

    # Calcola la pendenza (m)
    m = (y2 - y1) / (x2 - x1)

    # Calcola l'intercetta (b)
    b = y1 - m * x1

    return np.rad2deg(np.arctan(m)), b


def point_distance(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    distance = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
    return distance


def angle_between_lines(m1, m2):
    # Step 3: Calculate the angle in radians
    theta_radians = np.arctan(abs((m1 - m2) / (1 + m1 * m2)))

    # Step 4: Convert the angle from radians to degrees
    theta_degrees = np.degrees(theta_radians)

    return theta_degrees


def find_point_on_line(x0, y0, m, L):
    """
    Trova le coordinate del punto a distanza L dal punto (x0, y0) su una retta con coefficiente angolare m.

    Parametri:
    x0 (float): Coordinata x del punto A.
    y0 (float): Coordinata y del punto A.
    m (float): Coefficiente angolare della retta.
    L (float): Distanza tra il punto A e il punto cercato B.

    Ritorna:
    tuple: Coordinate del punto B (x1, y1).
    """
    delta_x = L / np.sqrt(1 + m ** 2)
    delta_y = m * delta_x

    x1 = x0 + delta_x
    y1 = y0 + delta_y

    return (x1, y1)

'''
    def calc_arm_rotation(self, person_point, arm):
        # d1 dist punto noto / persona; d2 distanza punto incognita / persona
        # l1 e l2 le linee che passano per i 2 punti
        # theta0 angolo interno
        # theta1 angolo finale

        # distanza tra punto noto e persona
        d1 = numpy.linalg.norm(person_point - arm[0].anchor_point)
        l1_m, l1_q = line_eq(person_point, arm[0].anchor_point)

        d2 = 100  # distanza minima persona/snodo
        l2_m, l2_q = line_eq(person_point, arm[1].free_point)

        theta0 = angle_between_lines(l1_m, l2_m)

        # lunghezza del segmento
        c = arm[0].length
        # Calcola l'angolo opposto al lato a
        cos_alpha = (d2 ** 2 + c ** 2 - d1 ** 2) / (2 * d2 * c)
        alpha_radians = np.arccos(cos_alpha)
        alpha_degrees = np.degrees(alpha_radians)

'''
