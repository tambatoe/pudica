import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf


def df_to_ds(dataframe_in):
    df_copy = dataframe_in.copy()
    labels = df_copy.pop('category')

    ds_tuple = (np.asarray(df_copy), labels.to_list())
    return tf.data.Dataset.from_tensor_slices(ds_tuple).batch(32)


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')

    df_c = pd.read_csv("data/close.csv")
    df_o = pd.read_csv("data/open.csv")
    df_c['category'] = 0
    df_o['category'] = 1

    New_df = pd.concat([df_c, df_o])

    train, test = train_test_split(New_df, test_size=0.2)
    test, val = train_test_split(test, test_size=0.4)

    train_ds = df_to_ds(train)
    test_ds = df_to_ds(test)
    val_ds = df_to_ds(val)

    for item in train_ds.take(2):
        print(item)

    input_shape = (34, 1)

    model = tf.keras.Sequential([
        # tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=input_shape),
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    model.fit(train_ds,
              validation_data=test_ds,
              epochs=100)

    evaluation = model.evaluate(val_ds)
    print(evaluation)
