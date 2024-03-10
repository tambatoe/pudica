import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf


def line_to_matrix(line, label):
    row = np.zeros((line.shape[0] // 2, 2), np.float32)
    x_val = np.take(line, list(range(0, line.shape[0], 2)))
    y_val = np.take(line, list(range(1, line.shape[0], 2)))

    for i in range(0, row.shape[0]):
        row[i] = [x_val[i], y_val[i]]

    return [row, label]


def df_to_ds(dataframe_in):
    df_copy = dataframe_in.copy()
    labels = df_copy.pop(1)
    df_np = df_copy.to_numpy()
    df_list = []

    for el in df_np:
        df_list.append(el[0])

    return tf.data.Dataset.from_tensor_slices((df_list, labels.to_list())).batch(32)


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')

    df_c = pd.read_csv("data/close.csv")
    df_o = pd.read_csv("data/open.csv")

    dataset_list = []

    for line in df_c.to_numpy():
        dataset_list.append(line_to_matrix(line, 1))

    for line in df_o.to_numpy():
        dataset_list.append(line_to_matrix(line, 2))

    New_df = shuffle(pd.DataFrame(dataset_list))

    train, test = train_test_split(New_df, test_size=0.2)
    test, val = train_test_split(test, test_size=0.4)

    train_ds = df_to_ds(train)
    test_ds = df_to_ds(test)
    val_ds = df_to_ds(val)

    for item in train_ds.take(2):
        print(item)

    input_shape = (17, 2)

    model = tf.keras.Sequential([
        # tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=test_ds,
              epochs=100)

    evaluation = model.evaluate(val_ds)
    print(evaluation)
