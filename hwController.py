from functools import partial

import numpy
import numpy as np

from hw_math_support import *


class SingleArm:

    def __init__(self, origin, length):
        origin = np.array(origin)
        self.origin = origin
        self.length = length
        self.theta = 0
        self.point1 = origin + [0, length]

        self.min_distance = 10

    def rotate(self, person_point):
        ''' dati
             pa = anchor point (origin)
             pl = punto libero
             pe = punto persona

             AE = distanza anchor persona
             LE = distanza persona punto libero pari alla distanza minima persona - braccio
             AL = lunghezza segmento
        '''

        # retta passante per pa - pe il valore di m in gradi servirà per ottenere l'angolo totale del braccio
        m, q = line_eq(self.origin, person_point)
        LE = 3 # 0.05
        AE = 3 # point_distance(self.origin, person_point)
        AL = 3 # 0.3 # self.length

        D = AE ** 2 + AL ** 2 - LE ** 2
        d = (2 * AE * AL)
        A = np.rad2deg(np.arccos(D / d))

        self.theta = A + m

        p1 = find_point_on_line(self.origin[0], self.origin[1], m, self.length)
        self.point1 = np.array(p1)

        return self.theta, self.point1


class PudicaStructure:

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

    def update_rotations(self, person_point):
        for i in range(0, 3):
            self.arms_left[i].rotate(person_point)
            self.arms_right[i].rotate(person_point)


class HwController:
    def __init__(self):
        self.person_bbox_left = None
        self.person_bbox_right = None
        self.open_percent_left = 100
        self.open_percent_right = 100
        self.limit_left = 0
        self.limit_right = 1

        self.structure = PudicaStructure()

    def add_detection(self, person_p0, person_p1, operation):
        self.person_bbox_left = np.array(person_p0)
        self.person_bbox_right = np.array(person_p1)

        if operation > 0.5:
            print('opening')

        center = (self.person_bbox_left + self.person_bbox_right) / 2

        self.structure.update_rotations(center)

    def move_physical(self):
        # deform points
        # estimate point in real space
        # calculate points position
        # apply points
        # send command to the hardware actuator controller
        pass


if __name__ == '__main__':
    controller = HwController()

    controller.add_detection([0.49, 0.5], [0.51, 0.5], 0.3)
