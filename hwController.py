import numpy
import numpy as np


class SingleArm:
    '''
    p_anchor è il punto di ancoraggio, può essere fisso o l'estremità di un altro braccio.
    '''

    def __init__(self, p_anchor, length):
        self.anchor_point = p_anchor
        self.theta = 0
        self.free_point = p_anchor
        self.length = length
        self.free_point[1] += length


def line_eq(point1, point2):
    # Calculate the slope (m)
    m = (point2[1] - point1[1]) / (point2[0] - point1[0])
    # Calculate the y-intercept (b)
    b = point1[1] - m * point1[0]
    return m, b


def calculate_angle_between_lines(m1, m2):
    # Step 3: Calculate the angle in radians
    theta_radians = np.arctan(abs((m1 - m2) / (1 + m1 * m2)))

    # Step 4: Convert the angle from radians to degrees
    theta_degrees = np.degrees(theta_radians)

    return theta_degrees


class PudicaStructure:

    def calc_arm_rotation(self, person_point, arm):
        # d1 dist punto noto / persona; d2 distanza punto incognita / persona
        # l1 e l2 le linee che passano per i 2 punti
        # theta0 angolo interno
        # theta1 angolo finale

        # distanza tra punto noto e persona
        d1 = numpy.linalg.norm(person_point - arm[0].anchor_point)
        l1_m, l1_q = line_eq(person_point, arm[0].anchor_point)

        d2 = 100 # distanza minima persona/snodo
        l2_m, l2_q = line_eq(person_point, arm[1].free_point)

        theta0 = calculate_angle_between_lines(l1_m, l2_m)

        # lunghezza del segmento
        c = arm[0].length
        # Calcola l'angolo opposto al lato a
        cos_alpha = (d2 ** 2 + c ** 2 - d1 ** 2) / (2 * d2 * c)
        alpha_radians = np.arccos(cos_alpha)
        alpha_degrees = np.degrees(alpha_radians)



    def calc_rotations(self, person_point):
        self.calc_arm_rotation(person_point, self.arms_left)

    def __init__(self):

        l1 = 88.59
        l2 = 122.82
        l3 = 88.59

        self.p0_0 = [0, 0]
        self.p0_0 = [0, (l1 + l2 + l3)]

        self.arms_left = [SingleArm([0, 0], l1),
                          SingleArm([l1, 0], l2),
                          SingleArm([l1 + l2, 0], l3)]

        self.arms_right = [SingleArm([0, 0], l1),
                           SingleArm([l1, 0], l2),
                           SingleArm([l1 + l2, 0], l3)]



class HwController:
    def __init__(self):
        self.open_percent_left = 100
        self.open_percent_right = 100
        self.limit_left = 0
        self.limit_right = 1

    def add_detection(self, p0, p1, operation):
        self.limit_left = p0[0]
        self.limit_right = p1[0]

        if operation > 0.5:
            print('opening')

    def actuate(self):
        # deform points
        # estimate point in real space
        # calculate points position
        # apply points
        # send command to the hardware actuator controller
        pass


if __name__ == '__main__':
    # Esempio di utilizzo:
    sa = SingleArm([1, 1], 5)
    sa.rotate([4, 4])
    print("Punto libero ruotato:", sa.free_point)
