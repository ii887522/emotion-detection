import common
from keras.models import Sequential
from keras.layers import Dense, Reshape, BatchNormalization, Dropout, Conv2D, Activation, MaxPooling2D, Flatten, PReLU
import keras.utils as keras_utils
from keras.regularizers import L2
from keras import models
from keras.constraints import MaxNorm
import numpy as np
from typing import Tuple
import sklearn.utils as sklearn_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import keras_tuner as kt


INPUT_DIR_PATH = "res/emotions"
IMAGE_SIZE = (48, 48)
MODEL_FILE_PATH = "build/emotion_model.h5"

LABEL_MAP = {
    "Angry": 0,
    "Fear": 1,
    "Happy": 2,
    "Neutral": 3,
    "Sad": 4,
    "Surprise": 5,
}


def main():
    # Load datasets
    print("Loading training datasets...")
    (train_xs, train_ys) = load_emotion_datasets("Training")
    print("Loading test datasets...")
    (test_xs, test_ys) = load_emotion_datasets("Testing")

    # Build a neural network
    model: Sequential

    if os.path.isfile(MODEL_FILE_PATH):
        model = models.load_model(MODEL_FILE_PATH)

    else:
        # Build a hyperparameter searcher
        tuner = kt.Hyperband(make_ann, "val_accuracy", factor=2, directory="build", project_name="keras_tuner")

        # Search the best hyperparameters
        tuner.search(
            x=train_xs,
            y=keras_utils.to_categorical(train_ys, len(LABEL_MAP)),
            batch_size=32,
            epochs=100,
            validation_data=(test_xs, keras_utils.to_categorical(test_ys, len(LABEL_MAP))),
            class_weight=dict(enumerate(sklearn_utils.compute_class_weight("balanced", classes=np.unique(train_ys), y=train_ys), 0)),
            callbacks=[EarlyStopping(monitor="val_accuracy", patience=100)]
        )

        best_hps = tuner.get_best_hyperparameters()[0]

        print(f"""
The hyperparameter search is complete. The optimal hyperparameters are:
- input_dropout_rate: {best_hps.get("input_dropout_rate")}

- block_1_units: {best_hps.get("block_1_units")}
- block_1_max_norm: {best_hps.get("block_1_max_norm")}
- block_1_dropout_rate: {best_hps.get("block_1_dropout_rate")}

- block_2_units: {best_hps.get("block_2_units")}
- block_2_max_norm: {best_hps.get("block_2_max_norm")}
- block_2_dropout_rate: {best_hps.get("block_2_dropout_rate")}

- block_3_units: {best_hps.get("block_3_units")}
- block_3_max_norm: {best_hps.get("block_3_max_norm")}
- block_3_dropout_rate: {best_hps.get("block_3_dropout_rate")}

- block_4_units: {best_hps.get("block_4_units")}
- block_4_max_norm: {best_hps.get("block_4_max_norm")}
- block_4_dropout_rate: {best_hps.get("block_4_dropout_rate")}
""")

        model = tuner.hypermodel.build(best_hps)

    # Train the neural network
    history = model.fit(
        x=train_xs,
        y=keras_utils.to_categorical(train_ys, len(LABEL_MAP)),
        batch_size=32,
        epochs=100,
        validation_data=(test_xs, keras_utils.to_categorical(test_ys, len(LABEL_MAP))),
        class_weight=dict(enumerate(sklearn_utils.compute_class_weight("balanced", classes=np.unique(train_ys), y=train_ys), 0)),
        callbacks=[
            ModelCheckpoint(filepath=MODEL_FILE_PATH, monitor="val_accuracy", save_best_only=True),
            EarlyStopping(monitor="val_accuracy", patience=100)
        ]
    )
    print(history.history)


def make_cnn(hp: kt.HyperParameters):
    model = Sequential()

    # Block-1
    model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',kernel_constraint=MaxNorm(3),input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(32,(3,3),padding='same',kernel_initializer='he_normal',kernel_constraint=MaxNorm(3),input_shape=(IMAGE_SIZE[0],IMAGE_SIZE[1],1)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    # Block-2
    model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal',kernel_constraint=MaxNorm(3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64,(3,3),padding='same',kernel_initializer='he_normal',kernel_constraint=MaxNorm(3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    # Block-3
    model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal',kernel_constraint=MaxNorm(3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128,(3,3),padding='same',kernel_initializer='he_normal',kernel_constraint=MaxNorm(3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    # Block-4
    model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal',kernel_constraint=MaxNorm(3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256,(3,3),padding='same',kernel_initializer='he_normal',kernel_constraint=MaxNorm(3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    # Block-5
    model.add(Flatten())
    model.add(Dense(64,kernel_initializer='he_normal',kernel_constraint=MaxNorm(3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Block-6
    model.add(Dense(64,kernel_initializer='he_normal',kernel_constraint=MaxNorm(3)))
    model.add(Activation('elu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    # Block-7
    model.add(Dense(len(LABEL_MAP),kernel_initializer='glorot_normal'))
    model.add(Activation('softmax'))

    model.summary()
    model.compile("adam", "categorical_crossentropy", ["accuracy"])

    return model


def make_ann(hp: kt.HyperParameters):
    model = Sequential()
    model.add(
        Reshape(
            target_shape=(IMAGE_SIZE[0] * IMAGE_SIZE[1],),
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)
        )
    )
    model.add(Dropout(hp.Float("input_dropout_rate", 0.0, 0.9, 0.1, default=0.2)))

    # Block-1
    model.add(
        Dense(
            units=hp.Int("block_1_units", 1, 1024, 1, default=512),
            activation="selu",
            kernel_initializer="lecun_normal",
            # kernel_regularizer=L2(0.001),
            # bias_regularizer=L2(0.001),
            kernel_constraint=MaxNorm(hp.Int("block_1_max_norm", 3, 5, 1, default=3))
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float("block_1_dropout_rate", 0.0, 0.9, 0.1, default=0.2)))

    # Block-2
    model.add(
        Dense(
            units=hp.Int("block_2_units", 1, 1024, 1, default=512),
            activation="selu",
            kernel_initializer="lecun_normal",
            # kernel_regularizer=L2(0.001),
            # bias_regularizer=L2(0.001),
            kernel_constraint=MaxNorm(hp.Int("block_2_max_norm", 3, 5, 1, default=3))
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float("block_2_dropout_rate", 0.0, 0.9, 0.1, default=0.2)))

    # Block-3
    model.add(
        Dense(
            units=hp.Int("block_3_units", 1, 1024, 1, default=512),
            activation="selu",
            kernel_initializer="lecun_normal",
            # kernel_regularizer=L2(0.001),
            # bias_regularizer=L2(0.001),
            kernel_constraint=MaxNorm(hp.Int("block_3_max_norm", 3, 5, 1, default=3))
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float("block_3_dropout_rate", 0.0, 0.9, 0.1, default=0.2)))

    # Block-4
    model.add(
        Dense(
            units=hp.Int("block_4_units", 1, 1024, 1, default=512),
            activation="selu",
            kernel_initializer="lecun_normal",
            # kernel_regularizer=L2(0.001),
            # bias_regularizer=L2(0.001),
            kernel_constraint=MaxNorm(hp.Int("block_4_max_norm", 3, 5, 1, default=3))
        )
    )
    model.add(BatchNormalization())
    model.add(Dropout(hp.Float("block_4_dropout_rate", 0.0, 0.9, 0.1, default=0.2)))

    # Block-5
    model.add(
        Dense(
            units=len(LABEL_MAP),
            activation="softmax",
            kernel_initializer="glorot_normal",
        )
    )

    model.summary()
    model.compile("adam", "categorical_crossentropy", ["accuracy"])

    return model


def load_emotion_datasets(subset_name: str) -> Tuple[np.ndarray, np.ndarray]: # ((N, 48, 48, 1), (N,))
    (angry_images, angry_labels) = load_emotion_dataset(subset_name=subset_name, emotion_name="Angry")
    (fear_images, fear_labels) = load_emotion_dataset(subset_name=subset_name, emotion_name="Fear")
    (happy_images, happy_labels) = load_emotion_dataset(subset_name=subset_name, emotion_name="Happy")
    (neutral_images, neutral_labels) = load_emotion_dataset(subset_name=subset_name, emotion_name="Neutral")
    (sad_images, sad_labels) = load_emotion_dataset(subset_name=subset_name, emotion_name="Sad")
    (surprise_images, surprise_labels) = load_emotion_dataset(subset_name=subset_name, emotion_name="Surprise")

    return (
        np.concatenate((angry_images, fear_images, happy_images, neutral_images, sad_images, surprise_images)),
        np.concatenate((angry_labels, fear_labels, happy_labels, neutral_labels, sad_labels, surprise_labels)),
    )


def load_emotion_dataset(subset_name: str, emotion_name: str) -> Tuple[np.ndarray, np.ndarray]: # ((N, 48, 48, 1), (N,))
    images = common.load_grayscale_images(dir_path=f"{INPUT_DIR_PATH}/{subset_name}/{emotion_name}")
    labels = np.array([LABEL_MAP.get(emotion_name, "")] * images.shape[0])
    return (images, labels)


if __name__ == "__main__":
    main()
