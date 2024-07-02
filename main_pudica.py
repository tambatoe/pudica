import time

from network_detection import Detection
import cv2
import numpy as np
import tensorflow as tf

import queue
from threading import Thread
from hwController import HwController


def capture_f():
    global do_loop
    global capture_inference_q
    global last_frame_q

    vid = cv2.VideoCapture(0)

    i = 0
    while do_loop:
        ret, frame = vid.read()
        time.sleep(0.03)
        try:
            # TODO resize here, reduce memory footprint
            capture_inference_q.put(frame, block=False)
            last_frame_q.put(frame, block=False)
        except:
            # print("skip frame enqueue ", i)
            i += 1

    vid.release()


def pose_detect_f():
    global do_loop
    global capture_inference_q
    global inference_pose_q

    detection = Detection()
    print("detection start")
    i = 0
    while do_loop:
        frame = capture_inference_q.get(block=True, timeout=None)
        detection.set_image(frame)
        kp = detection.predict()
        try:
            # TODO resize here, reduce memory footprint
            inference_pose_q.put(kp, block=False)
        except:
            # print("skipped detect results ", i)
            i += 1


def pose_eval_f():
    global do_loop
    global inference_pose_q
    global pose_resultev_q

    print("eval start")

    classification = tf.keras.models.load_model('weights.best.hdf5')

    i = 0
    while do_loop:
        kp = inference_pose_q.get(block=True, timeout=None)
        reshaped_array = np.expand_dims(np.asarray(kp).flatten(), axis=0)
        result = classification.predict(reshaped_array, verbose=0)

        try:
            # TODO resize here, reduce memory footprint
            pose_resultev_q.put([kp, result], block=False)
        except:
            # print("skip frame enqueue ", i)
            i += 1


def loop_phisical_movement(hw_controller):
    global do_loop

    while do_loop:
        hw_controller.move_physical()


def result_evaluation_f():
    global do_loop
    global pose_resultev_q
    global last_frame_q

    meter_size = [16, 9]

    print("result eval start")

    b = 0
    g = 0
    r = 0

    operation_avg = 0.5

    center_list = []
    hw_controller = HwController()

    thread = Thread(target=loop_phisical_movement, args=(hw_controller,))
    thread.start()

    while do_loop:
        kp, result = pose_resultev_q.get(block=True, timeout=None)

        # operation = np.argmax(result[0], axis=0)
        operation_avg += 0.05 if result[0][0] < result[0][1] else -0.05
        operation_avg = np.clip(operation_avg, 0, 1)

        kpma = np.ma.masked_where(kp == -1, kp)
        min_y = np.min(kpma[:, 0], axis=0)
        max_y = np.max(kp[:, 0], axis=0)
        min_x = np.min(kpma[:, 1], axis=0)
        max_x = np.max(kp[:, 1], axis=0)

        p0 = (np.asarray([min_x, min_y]) - 0.5) * meter_size  # normalize center of image
        p1 = (np.asarray([max_x, max_y]) - 0.5) * meter_size

        hw_controller.add_detection(p0, p1, operation_avg)

        # print(f"operation {operation_avg}")

        try:
            frame = last_frame_q.get(block=False, timeout=1)
            height, width, _ = frame.shape

            center_p = np.array([width * (min_x + max_x) / 2, height * (min_y + max_y) / 2])
            center_list.append(center_p)
            if len(center_list) > 60:
                center_list.pop(0)

            center = np.average(center_list, axis=0)

            g = g + 2 if operation_avg > 0.5 else g - 2
            g = min(255, max(0, g))
            r = 255 - g

            cv2.circle(frame, center.astype(np.int32), 20, (b, g, r), -1)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", frame)
            cv2.waitKey(2)

        except:
            print("cant dequeue")
        # print(p0, p1)


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')

    capture_inference_q = queue.Queue(maxsize=3)
    inference_pose_q = queue.Queue(maxsize=3)
    pose_resultev_q = queue.Queue(maxsize=3)
    last_frame_q = queue.Queue(maxsize=3)

    do_loop = True

    capture_t = Thread(target=capture_f)
    pose_detect_t = Thread(target=pose_detect_f)
    pose_eval_t = Thread(target=pose_eval_f)
    result_evaluation_t = Thread(target=result_evaluation_f)

    capture_t.start()
    pose_detect_t.start()
    pose_eval_t.start()
    result_evaluation_t.start()

    # pose_eval_t.join()
    while do_loop:
        time.sleep(1)

        # print("cose")
    #     t0 = time.time()
    #     # Capture the video frame
    #     # by frame
    #
    #     # Display the resulting frame
    #     cv2.imshow('frame', frame)
    #
    #     # the 'q' button is set as the
    #     # quitting button you may use any
    #     # desired button of your choice
    #     # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     #     do_loop = False
    #     #     break
    #
    #     # print(time.time() - t0)
    #
    # # Destroy all the windows
    # # cv2.destroyAllWindows()
