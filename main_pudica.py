import time

from network_detection import Detection
import cv2
import numpy as np
import tensorflow as tf


# todo: add producer consumer for capture (wasting 300% of time atm)

if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')

    detection = Detection()
    classification = tf.keras.models.load_model('saved_models/classification')

    vid = cv2.VideoCapture(0)

    do_loop = True
    while do_loop:
        t0 = time.time()
        # Capture the video frame
        # by frame
        ret, frame = vid.read()
        detection.set_image(frame)
        kp = detection.predict()
        normalized = detection.normalize_points(kp)
        reshaped_array = np.asarray(normalized).reshape(-1, 17 * 2)

        result = classification.predict(reshaped_array)
        print(result)


        # Display the resulting frame
        cv2.imshow('frame', frame)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            do_loop = False
            break

        # print(time.time() - t0)

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
