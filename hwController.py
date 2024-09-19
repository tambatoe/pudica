import json
import math
import time
from math import degrees
import serial

import numpy as np

'''
command 1 STOP, 0, 0
command 2 speed, motor id, new speed
command 3 move, motor id, position
command 4 get movement status, motor id, 
            response: 0 moving, 1 done
'''


def find_tangent_angles(x, y, x1, y1, r):
    # Step 1: Calculate the angle to the center of the circle
    theta_center = np.arctan2(y1 - y, x1 - x)

    # Step 2: Calculate the distance from the origin to the center of the circle
    d = np.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)

    # Step 3: Check if tangents exist (distance must be greater than the radius)
    if d <= r:
        raise ValueError("The origin is inside the circle or on the circle, no tangents exist.")

    # Step 4: Calculate the angle between the center line and the tangent lines
    alpha = np.arcsin(r / d)

    # Step 5: Calculate the two tangent angles
    theta_tangent_1 = theta_center + alpha
    theta_tangent_2 = theta_center - alpha

    return np.degrees(theta_tangent_1), np.degrees(theta_tangent_2)


def get_vector_endpoint(x0, y0, length, angle_deg):
    # Convert angle from degrees to radians
    angle_rad = np.radians(angle_deg)

    # Calculate the x and y components of the vector
    x_end = x0 + length * np.cos(angle_rad)
    y_end = y0 + length * np.sin(angle_rad)

    return x_end, y_end


class UART:
    ser = None

    @staticmethod
    def init(port='/dev/ttyUSB0', baudrate=9600, timeout=1):
        """ Initialize the UART connection if it's not already initialized """

        return 0
        if UART.ser is None:
            UART.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=timeout
            )
            if UART.ser.is_open:
                print(f"Connected to {UART.ser.name}")

    @staticmethod
    def send_command(command, param0=0, param1=0):
        max_bytes = 100
        """ Send a UART command """
        print(f'sending command {command} {param0} {param1}')

        # Convert the command to bytes and send it
        if isinstance(command, str):
            command = command.encode('utf-8')  # Convert string to bytes if necessary
        if isinstance(param0, str):
            param0 = param0.encode('utf-8')
        if isinstance(param1, str):
            param1 = param1.encode('utf-8')

        if UART.ser is None:
            print(f"Command sent: {command} {param0} {param1}")
        else:
            UART.ser.write(command)
            UART.ser.write(param0)
            UART.ser.write(param1)

    @staticmethod
    def read_response(max_bytes=100):
        """ Read a response from UART """
        if UART.ser is None:
            print('responding 0')
            return 0
        # Read response from UART
        response = UART.ser.read(max_bytes)  # Read up to max_bytes or until timeout
        return response.decode('utf-8')  # Convert bytes to string

    @staticmethod
    def close():
        """ Close the UART connection """
        if UART.ser is not None:
            UART.ser.close()
            UART.ser = None
            print("UART connection closed.")


class PudicaStructure:
    def __init__(self):
        self.l1 = 88.59
        self.l2 = 122.82
        self.l3 = 88.59

        self.left_rots = [0, 0, 0, 0]
        self.right_rots = [0, 0, 0, 0]

        self.current_step = 0
        self.last_automation = 'breath'

        with open('hwController_movements.json') as f:
            self.automation = json.load(f)

    def update_on_point(self, person_point, person_radius):
        # aggiorna le posizioni sfruttando il punto centrale (questo per la parte interattiva)

        x, y = 0, 0  # Origine del primo segmento
        tangent_angles = find_tangent_angles(x, y, person_point[0], person_point[1], person_radius * 1.1)
        self.left_rots[0] = tangent_angles[0]

        x1, y1 = get_vector_endpoint(x, y, self.l1, tangent_angles[0])
        tangent_angles = find_tangent_angles(x1, y1, person_point[0], person_point[1], person_radius * 1.1)
        self.left_rots[1] = tangent_angles[0]

        x2, y2 = get_vector_endpoint(x1, y1, self.l1, tangent_angles[0])
        tangent_angles = find_tangent_angles(x2, y2, person_point[0], person_point[1], person_radius * 1.1)
        self.left_rots[3] = tangent_angles[0]

        x, y = 1, 1  # Origine del primo segmento
        tangent_angles = find_tangent_angles(x, y, person_point[0], person_point[1], person_radius * 1.1)
        self.right_rots[0] = tangent_angles[1]

        x1, y1 = get_vector_endpoint(x, y, self.l1, tangent_angles[1])
        tangent_angles = find_tangent_angles(x1, y1, person_point[0], person_point[1], person_radius * 1.1)
        self.right_rots[1] = tangent_angles[1]

        x2, y2 = get_vector_endpoint(x1, y1, self.l1, tangent_angles[1])
        tangent_angles = find_tangent_angles(x2, y2, person_point[0], person_point[1], person_radius * 1.1)
        self.right_rots[3] = tangent_angles[1]

        print(f"The angles are: {self.left_rots} {self.right_rots}")

    def update_on_deg(self, automation_type, current_motor_status):
        # aggiorna le posizioni applicando direttamente gli angoli in degrees (questo per le sequenze automatiche)

        if self.last_automation == automation_type:
            """ valuto se lo step è stato completato e devo andare al successivo o sta ancora ruotando"""

            go_next = all(element == 0 for element in current_motor_status)
            if go_next:
                self.current_step = (self.current_step + 1) % len(self.automation['left'][automation_type])
        else:
            """se ho cambiato tipo di automazione parto dallo step 0"""
            self.current_step = 0

        self.left_rots = self.automation['left'][automation_type][self.current_step]
        self.right_rots = self.automation['right'][automation_type][self.current_step]

    def next_deg(self, current_motor_status, automation_type, person_point, person_radius):
        """
        ritorna il prossimo angolo da raggiungere.
        Può essere il punto ideale in caso di punto centrale o il passo finale da raggiungere se automazione
        :return:  2 array di gradi destra e sinistra
        """

        if automation_type == 'follow_person':
            self.update_on_point(person_point, person_radius)
        else:
            self.update_on_deg(automation_type, current_motor_status)
        return self.left_rots, self.right_rots


class HwController:
    """
    Modello del controller fisico, decide la posizione degli attutatori in base a operazione corrente
    (apri/chiudi) e alla posizione della persona nella scena.
    """

    def __init__(self):
        """
        init
        """
        self.person_bbox_left = None
        self.person_bbox_right = None

        self.operation = 1.0
        self.center = None
        self.circle_radius = 0

        # emergency stop
        self.stop = False

        self.prev_automation_type = 'breath'
        self.pudica = PudicaStructure()
        UART.init()

    def add_detection(self, person_p0, person_p1, operation):
        """
        Aggiorna posizioni e operazioni delle persone.
        :param person_p0: bounding box della persona, top left
        :param person_p1: bounding box della persona, bottom right
        :param operation: valutazione apri/chiudi
        :return:
        """

        self.person_bbox_left = np.array(person_p0)
        self.person_bbox_right = np.array(person_p1)

        self.operation = operation
        # TODO: valutare qua se op = 0 è apri o chiudi
        if operation > 0.5:
            print('opening')

        self.center = (self.person_bbox_left + self.person_bbox_right) / 2

        x1, y1 = self.person_bbox_left
        x2, y2 = self.person_bbox_right

        diagonal_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        self.circle_radius = diagonal_length / 2

    def move_physical(self):
        time.sleep(0.003)

        if self.stop:
            UART.send_command(1, 0x00, 0x00)
            return

        # TODO: mappatura dei vari motori posizione - id

        current_status = []
        # lettura pos correnti
        for i in range(1, 11):
            UART.send_command(4, i, 0)  # 2 motori all'inizio
            current_status.append(UART.read_response())

        # TODO decidere automation type in base alla presenza e posizione di una persona

        automation_type = ''
        person_in_scene = 0.2 < self.center[0] < 0.8 and 0.2 < self.center[1] < 0.8
        if self.prev_automation_type == 'follow_person':
            if person_in_scene:
                automation_type = 'follow_person'
            else:
                automation_type = 'closing'
        elif self.prev_automation_type == 'closing':
            if all(element == 0 for element in current_status):
                automation_type = 'breath'
            else:
                automation_type = 'closing'
        elif self.prev_automation_type == 'opening':
            if all(element == 0 for element in current_status):
                automation_type = 'follow_person'
            else:
                automation_type = 'opening'
        elif self.prev_automation_type == 'breath':
            if person_in_scene:
                automation_type = 'opening'
            else:
                automation_type = 'breath'

        self.prev_automation_type = automation_type

        degs_l, degs_r = self.pudica.next_deg(current_status, automation_type,
                                              self.center, self.circle_radius)

        # move, motor id, degree
        UART.send_command(1, 0x01, degs_l[0])  # 2 motori all'inizio
        UART.send_command(1, 0x02, degs_l[0])
        UART.send_command(1, 0x03, degs_l[1])  # 1 motore centrale
        UART.send_command(1, 0x04, degs_l[2])  ## questo è il diagonale
        UART.send_command(1, 0x05, degs_l[3])  # 1 motore alla fine

        UART.send_command(1, 0x06, degs_r[0])
        UART.send_command(1, 0x07, degs_r[0])
        UART.send_command(1, 0x08, degs_r[1])
        UART.send_command(1, 0x09, degs_r[2])
        UART.send_command(1, 0x0A, degs_r[3])
