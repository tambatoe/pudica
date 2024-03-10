import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

if __name__ == '__main__':
    print('ciao')

    tf.config.set_visible_devices([], 'GPU')

    df_c = pd.read_csv("../data/close.csv")
    df_o = pd.read_csv("../data/close.csv")

    df_c.insert(0, "category", "close", True)
    df_o.insert(0, "category", "open", True)

    df_merge = pd.concat([df_o, df_c])

    df_merge['category'] = df_merge['category'].astype('category')
    df_merge['cat_new'] = df_merge['category'].cat.codes

    # Create an instance of One-hot-encoder
    enc = OneHotEncoder()
    enc_data = pd.DataFrame(enc.fit_transform(
        df_merge[['cat_new']]).toarray())

    # Merge with main
    New_df = df_merge.join(enc_data)
    New_df = New_df.drop(['category', 0, 1], axis=1)
    New_df = shuffle(New_df)
    print(New_df)

    train, test = train_test_split(New_df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)

    train_labels = train.pop('cat_new')
    test_labels = test.pop('cat_new')
    val_labels = val.pop('cat_new')

    train_ds = tf.data.Dataset.from_tensor_slices((np.asarray(train), train_labels)).batch(32)
    test_ds = tf.data.Dataset.from_tensor_slices((np.asarray(test), test_labels)).batch(32)
    val_ds = tf.data.Dataset.from_tensor_slices((np.asarray(val), val_labels)).batch(32)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu',  input_shape=[None, 34]),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(2, activation='linear')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['binary_accuracy'])

    model.fit(train_ds,
              validation_data=test_ds,
              epochs=500000)

    evaluation = model.evaluate(val_ds)
    print(evaluation)

