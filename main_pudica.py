import time

from network_detection import Detection
import cv2
import numpy as np
import tensorflow as tf

import queue
from threading import Thread


def capture_f():
    global do_loop
    global capture_inference_q

    vid = cv2.VideoCapture(0)

    i = 0
    while do_loop:
        ret, frame = vid.read()

        try:
            # TODO resize here, reduce memory footprint
            capture_inference_q.put(frame, block=False)
        except:
            # print("skip frame enqueue ", i)
            i += 1

        # cv2.imshow("f", frame)
        # cv2.waitKey(10)

    vid.release()


def pose_detect_f():
    global do_loop
    global capture_inference_q
    global inference_pose_q

    detection = Detection()

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


def result_evaluation_f():
    global do_loop
    global pose_resultev_q

    meter_size = [16, 9]


    while do_loop:
        kp, result = pose_resultev_q.get(block=True, timeout=None)

        operation = np.argmax(result, axis=0)
        kpma = np.ma.masked_where(kp == -1, kp)
        min_x = np.min(kpma[:, 0], axis=0)
        max_x = np.max(kp[:, 0], axis=0)
        min_y = np.min(kpma[:, 1], axis=0)
        max_y = np.max(kp[:, 1], axis=0)

        p0 = (np.asarray([min_x, min_y]) - 0.5) * meter_size  # normalize center of image
        p1 = (np.asarray([max_x, max_y]) - 0.5) * meter_size

        print(p0, p1)


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')

    capture_inference_q = queue.Queue(maxsize=3)
    inference_pose_q = queue.Queue(maxsize=3)
    pose_resultev_q = queue.Queue(maxsize=3)

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
