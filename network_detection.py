from defines import KEYPOINT_DICT, column_names

import cv2

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


def get_center_point(p0, p1):
    return [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2]


class Detection:

    def __init__(self):
        self.model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        self.model = self.model.signatures['serving_default']

        self.image = None

    def set_image(self, img):
        input_size = 192
        m_img = tf.convert_to_tensor(img)
        m_image = tf.expand_dims(m_img, axis=0)
        m_image = tf.image.resize_with_pad(m_image, input_size, input_size)

        # SavedModel format expects tensor type of int32.
        self.image = tf.cast(m_image, dtype=tf.int32)

    def load_image(self, file_path):
        # TODO: aggiungere anche il modello di conversione immagine
        input_size = 192

        m_image = tf.io.read_file(file_path)
        m_image = tf.image.decode_jpeg(m_image)
        m_image = tf.expand_dims(m_image, axis=0)
        m_image = tf.image.resize_with_pad(m_image, input_size, input_size)

        # SavedModel format expects tensor type of int32.
        self.image = tf.cast(m_image, dtype=tf.int32)

    def normalize_points(self, keypoints):
        keypoints_n = []

        hips_center = get_center_point(keypoints[KEYPOINT_DICT['left_hip']], keypoints[KEYPOINT_DICT['right_hip']])
        shoulders_center = get_center_point(keypoints[KEYPOINT_DICT['left_shoulder']], keypoints[KEYPOINT_DICT['right_shoulder']])

        pose_center_new = get_center_point(keypoints[KEYPOINT_DICT['left_hip']], keypoints[KEYPOINT_DICT['right_hip']])

        torso_size = np.linalg.norm(np.asarray(shoulders_center) - np.asarray(hips_center))
        sz_multiplier = 2.5
        d = keypoints - pose_center_new
        max_dist = np.max(d)
        tszm = torso_size*sz_multiplier
        pose_size = 1 / max(tszm, max_dist)

        for kp in keypoints:
            new_kp = np.asarray([kp[0] - pose_center_new[0], kp[1] - pose_center_new[1]]) * pose_size
            keypoints_n.append(new_kp)

        return keypoints_n

    def predict(self):
        outputs = self.model(self.image)
        keypoints_with_scores = outputs['output_0'].numpy()[0][0]
        detection = []

        for kws in keypoints_with_scores:
            if kws[2] > 0.2:
                detection.append(kws[:2])
            else:
                detection.append([-1, -1])

        return np.asarray(detection)


'''
Main is for dataset prepare

https://www.kaggle.com/code/manaihamza/yogapose-classification

'''

if __name__ == '__main__':
    # import needed for train only
    tf.config.set_visible_devices([], 'GPU')

    import os
    import tqdm
    import pandas as pd

    detection_network = Detection()
    # model = load()

    # il main è usato per addestrare la rete.
    DATASET_ROOT = 'data'
    # otteniamo il nome delle classi, partendo dalla root del dataset
    classes = []
    for f in os.listdir(DATASET_ROOT):
        if os.path.isdir(os.path.join(DATASET_ROOT, f)):
            classes.append(f)

    print(f"classes {classes}")

    for m_class in classes:
        all_points = []
        class_root = os.path.join(DATASET_ROOT, m_class)
        print(f"processing {class_root}")

        for f in tqdm.tqdm(os.listdir(class_root)):
            detection_network.load_image(os.path.join(class_root, f))
            detected = detection_network.predict()
            normalized = detection_network.normalize_points(detected)
            all_points.append(normalized)

        if len(all_points) > 0:
            reshaped_array = np.asarray(all_points).reshape(-1, 17 * 2)
            points_df = pd.DataFrame(reshaped_array)
            points_df.columns = column_names
            points_df.to_csv(class_root + '.csv', index=False)
