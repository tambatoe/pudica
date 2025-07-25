import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

from classification_support import landmarks_to_embedding


def df_to_ds(dataframe_in, label_column_name):
    df_copy = dataframe_in.copy()
    labels = df_copy.pop(label_column_name)

    labels = tf.keras.utils.to_categorical(labels)

    ds_tuple = (np.asarray(df_copy), labels)
    return tf.data.Dataset.from_tensor_slices(ds_tuple).batch(32)


if __name__ == '__main__':
    tf.config.set_visible_devices([], 'GPU')

    df_c = pd.read_csv("/run/media/tambatoe/PC_STORAGE/Poses/close.csv")
    df_o = pd.read_csv("/run/media/tambatoe/PC_STORAGE/Poses/open.csv")
    df_c['category'] = 0
    df_o['category'] = 1

    New_df = pd.concat([df_c, df_o])
    New_df = shuffle(New_df)

    New_df['category'] = New_df['category'].astype('category')

    label_encoder = LabelEncoder()
    New_df['category_labels'] = label_encoder.fit_transform(New_df['category'])

    New_df.drop(["category"], axis=1, inplace=True)
    New_df.info()

    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)
    # print(New_df.head)

    # X = New_df.drop('category_labels', axis=1)
    # y = New_df['category_labels']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train, test = train_test_split(New_df, test_size=0.2)
    test, val = train_test_split(test, test_size=0.4)

    train_ds = df_to_ds(train, 'category_labels')
    test_ds = df_to_ds(test, 'category_labels')
    val_ds = df_to_ds(val, 'category_labels')

    # for item in train_ds.take(2):
    #     print(item)

    class_names = ['open', 'close']

    inputs = tf.keras.Input(shape=(51))
    embedding = landmarks_to_embedding(inputs)

    layer = keras.layers.Dense(128, activation=tf.nn.relu6)(embedding)
    layer = keras.layers.Dropout(0.5)(layer)
    layer = keras.layers.Dense(64, activation=tf.nn.relu6)(layer)
    layer = keras.layers.Dropout(0.5)(layer)
    outputs = keras.layers.Dense(len(class_names), activation="softmax")(layer)

    model = keras.Model(inputs, outputs)
    model.summary()

    initial_learning_rate = 0.0003
    total_epochs = 20000
    cosine_decay = tf.keras.experimental.CosineDecay(initial_learning_rate, total_epochs)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=cosine_decay)
    optimizer = tf.keras.optimizers.Adam(initial_learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Add a checkpoint callback to store the checkpoint that has the highest
    # validation accuracy.
    checkpoint_path = "weights.best.hdf5"
    checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 monitor='val_accuracy',
                                                 verbose=1,
                                                 save_best_only=True,
                                                 mode='max')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                  patience=50)

    model.fit(train_ds,
              epochs=total_epochs,
              batch_size=16,
              validation_data=test_ds,
              callbacks=[checkpoint,  earlystopping,
                         tf.keras.callbacks.TensorBoard()
                         ])

    model.save('saved_models/classification')

    evaluation = model.evaluate(val_ds)
    # print(evaluation)
