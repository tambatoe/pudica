from defines import KEYPOINT_DICT, column_names

import cv2

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub


class Detection:

    def __init__(self):
        # self.model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        # self.model = tf.keras.models.load_model('saved_models/movenet_light_4/saved_model.pb')
        # self.model = self.model.signatures['serving_default']

        self.model = tf.lite.Interpreter(model_path="saved_models/movenet_light_4/model.tflite")
        self.model.allocate_tensors()

        self.image = None

    def set_image(self, img):
        input_size = 192
        m_img = tf.convert_to_tensor(img)
        m_image = tf.expand_dims(m_img, axis=0)
        m_image = tf.image.resize_with_pad(m_image, input_size, input_size)
        self.image = m_image
        #
        # # SavedModel format expects tensor type of int32.
        # self.image = tf.cast(m_image, dtype=tf.int32)

    def load_image(self, file_path):
        # TODO: aggiungere anche il modello di conversione immagine
        input_size = 192

        m_image = tf.io.read_file(file_path)
        m_image = tf.image.decode_jpeg(m_image)
        m_image = tf.expand_dims(m_image, axis=0)
        m_image = tf.image.resize_with_pad(m_image, input_size, input_size)

        # SavedModel format expects tensor type of int32.
        self.image = tf.cast(m_image, dtype=tf.int32)

    def predict(self):
        # outputs = self.model(self.image)
        # keypoints_with_scores = outputs['output_0'].numpy()[0][0]

        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(self.image, dtype=tf.uint8)
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()
        self.model.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        self.model.invoke()
        # Get the model prediction.
        keypoints_with_scores = self.model.get_tensor(output_details[0]['index'])[0][0]

        detection = []

        for kws in keypoints_with_scores:
            if kws[2] > 0.2:
                detection.append(kws)
            else:
                detection.append([-1, -1, 0])

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
    DATASET_ROOT = '/run/media/tambatoe/PC_STORAGE/Poses/'
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
            # normalized = detection_network.normalize_points(detected)
            all_points.append(np.asarray(detected).flatten())

        if len(all_points) > 0:
            reshaped_array = np.asarray(all_points)
            points_df = pd.DataFrame(reshaped_array)
            # points_df.columns = column_names
            points_df.to_csv(class_root + '.csv', index=False)
