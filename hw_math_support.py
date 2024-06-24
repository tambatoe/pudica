import numpy as np


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


def angle_between_lines(m1, m2):
    # Step 3: Calculate the angle in radians
    theta_radians = np.arctan(abs((m1 - m2) / (1 + m1 * m2)))

    # Step 4: Convert the angle from radians to degrees
    theta_degrees = np.degrees(theta_radians)

    return theta_degrees


def calc_arm_rotation(center):
    return 0

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
