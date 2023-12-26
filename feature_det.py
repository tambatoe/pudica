"""
Feature detector

rete utilizzata per la feature detection. Il main si occupa del training
"""

## indice dei punti
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

column_names = [
    'nose_x', 'nose_y', 'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y',
    'left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y',
    'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y',
    'left_elbow_x', 'left_elbow_y', 'right_elbow_x', 'right_elbow_y',
    'left_wrist_x', 'left_wrist_y', 'right_wrist_x', 'right_wrist_y',
    'left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y',
    'left_knee_x', 'left_knee_y', 'right_knee_x', 'right_knee_y',
    'left_ankle_x', 'left_ankle_y', 'right_ankle_x', 'right_ankle_y'
]

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from utils import draw_detections

# Disable GPU devices
tf.config.set_visible_devices([], 'GPU')

module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
model = module.signatures['serving_default']
input_size = 192


def load_image(file_path):
    m_image = tf.io.read_file(file_path)
    m_image = tf.image.decode_jpeg(m_image)
    return m_image


if __name__ == '__main__':
    import os
    import tqdm
    import pandas as pd

    # il main è usato per addestrare la rete.
    DATASET_ROOT = 'data'
    show_debug = False
    # otteniamo il nome delle classi, partendo dalla root del dataset
    classes = []
    for f in os.listdir(DATASET_ROOT):
        if os.path.isdir(os.path.join(DATASET_ROOT, f)):
            classes.append(f)

    print(classes)

    for subdir in classes:
        all_points = []
        class_root = os.path.join(DATASET_ROOT, subdir)
        for f in tqdm.tqdm(os.listdir(class_root)):
            image = load_image(os.path.join(class_root, f))

            input_image = tf.expand_dims(image, axis=0)
            input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

            # SavedModel format expects tensor type of int32.
            input_image = tf.cast(input_image, dtype=tf.int32)
            # Run model inference.
            outputs = model(input_image)
            # Output is a [1, 1, 17, 3] tensor.
            keypoints_with_scores = outputs['output_0'].numpy()[0][0]

            # per ogni punto, rimuoviamo la detection
            detected = []

            for kws in keypoints_with_scores:
                if kws[2] > 0.2:
                    detected.append(kws[:2])
                else:
                    detected.append([-1, -1])

            detected = np.asarray(detected)
            if show_debug is True:
                draw_detections(os.path.join(class_root, f), detected)

            shoulder_l = detected[KEYPOINT_DICT['left_shoulder']]
            shoulder_r = detected[KEYPOINT_DICT['right_shoulder']]
            hip_l = detected[KEYPOINT_DICT['left_hip']]
            hip_r = detected[KEYPOINT_DICT['right_hip']]

            center = np.mean([
                shoulder_l, shoulder_r, hip_l, hip_r
            ], axis=0)

            delta_center = [0.5, 0.5] - center
            moved_center = np.asarray([d + delta_center for d in detected])

            # Clip values between 0 and 1
            moved_center_clipped = np.clip(moved_center, 0, 1)

            # Replace the specific value (-1, -1) back
            # importante conservare il -1 che è indicatore di scarsa accuracy/punto non identificato
            mask = (detected == [-1,
                                 -1])
            moved_center_clipped[mask] = -1
            moved_center = moved_center_clipped
            all_points.append(moved_center)

        if len(all_points) > 0:
            reshaped_array = np.asarray(all_points).reshape(2, 17*2)
            points_df = pd.DataFrame(reshaped_array)
            points_df.columns = column_names
            points_df.to_csv(class_root + '.csv', index=False)
            print(points_df.head())
            # all_points = np.asarray(all_points)
            # print (all_points.reshape(2, -1))
            # np.savetxt(class_root + ".csv", all_points.reshape(2, -1), delimiter=",")
