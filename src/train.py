import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Activation, SpatialDropout2D
from keras.constraints import MinMaxNorm
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, BackupAndRestore
from keras.optimizers import Adam
import sklearn.utils as sklearn_utils
import sklearn.metrics as sklearn_metrics
import csv
from itertools import islice
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import os


# File paths
EMOTION_DIR_PATH = "res/emotions"
BEST_CHECKPOINT_FILE_PATH = "build/best_model"
BEST_VALIDATION_ACCURACY_FILE_PATH = "build/best_val_acc.txt"
BACKUP_DIR_PATH = "tmp/backup"
REPORT_DIR_PATH = "reports"

LABELS = [
    "Neutral",
    "Happiness",
    "Surprise",
    "Sadness",
    "Anger",
    "Disgust",
    "Fear",
    "Contempt",
    "Unknown",
    "NF"
]


def main():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset()

    x_train = dataset["Training"]["x"]
    y_train = dataset["Training"]["y"]
    x_val = dataset["PublicTest"]["x"]
    y_val = dataset["PublicTest"]["y"]
    x_test = dataset["PrivateTest"]["x"]
    y_test = dataset["PrivateTest"]["y"]

    # Check the dataset shape
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)

    # Build a CNN
    model = make_cnn(input_shape=x_train.shape[1:], output_shape=(len(LABELS),))

    # Load the best validation accuracy
    try:
        print("Loading the best validation accuracy...")

        with open(BEST_VALIDATION_ACCURACY_FILE_PATH, "r") as best_val_acc_file:
            best_val_acc = float(best_val_acc_file.read())

        print("best_val_acc: ", best_val_acc)

    except:
        print("Failed to load the accuracy. Set best_val_acc = 0.0")
        best_val_acc = 0.0

    # Train the CNN
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=16,
        epochs=50,
        verbose=1,
        validation_data=(x_val, y_val),
        class_weight=dict(
            enumerate(
                sklearn_utils.compute_class_weight(
                    class_weight="balanced",
                    classes=np.unique(y_train),
                    y=y_train
                ),
                0
            )
        ),
        callbacks=[
            ModelCheckpoint(
                filepath=BEST_CHECKPOINT_FILE_PATH,
                monitor="val_accuracy",
                verbose=1,
                save_best_only=True,
                initial_value_threshold=best_val_acc
            ),
            BackupAndRestore(backup_dir=BACKUP_DIR_PATH),
            ReduceLROnPlateau(factor=1.0 / 3.0, verbose=1)
        ]
    )

    # Save the best validation accuracy
    print("Save the best validation accuracy...")

    with open(BEST_VALIDATION_ACCURACY_FILE_PATH, "w") as best_val_acc_file:
        best_val_acc_file.write(str(max(max(history.history['val_accuracy']), best_val_acc)))

    # Generate reports
    # Assume .gitkeep file always exist inside the reports folder
    current_run = len(os.listdir(REPORT_DIR_PATH))

    # Summarize history for accuracy and save the result
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f"{REPORT_DIR_PATH}/{current_run}/model-acc.png")

    # Summarize history for loss and save the result
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f"{REPORT_DIR_PATH}/{current_run}/model-loss.png")

    # Evaluate the CNN
    metrics = model.evaluate(x=x_test, y=y_test, verbose=1, return_dict=True)

    # Save the CNN evaluation result
    with open("{REPORT_DIR_PATH}/{current_run}/metrics.txt", "w") as metrics_file:
        metrics_file.write(str(metrics))

    # Predict from the CNN
    y_pred = model.predict(x=x_test, verbose=1)
    y_pred = np.argmax(y_pred, axis=1)

    # Plot a confusion matrix and save the result
    result = sklearn_metrics.confusion_matrix(y_test, y_pred, normalize="pred")
    df_cm = pd.DataFrame(result, index=LABELS, columns=LABELS)
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f"{REPORT_DIR_PATH}/{current_run}/confusion-matrix.png")


"""
{
    "Training": {
        "x": (N, 48, 48, 1),
        "y": (N,)
    },
    "PublicTest": {
        "x": (N, 48, 48, 1),
        "y": (N,)
    },
    "PrivateTest": {
        "x": (N, 48, 48, 1),
        "y": (N,)
    }
}
"""
def load_dataset() -> dict:
    input = {
        "Training": {},
        "PublicTest": {},
        "PrivateTest": {}
    }

    print("Loading dataset pixels...")
    pixels = load_dataset_pixels()
    input["Training"]["x"] = pixels["Training"]["x"]
    input["PublicTest"]["x"] = pixels["PublicTest"]["x"]
    input["PrivateTest"]["x"] = pixels["PrivateTest"]["x"]

    print("Loading dataset labels...")
    labels = load_dataset_labels()
    input["Training"]["y"] = labels["Training"]["y"]
    input["PublicTest"]["y"] = labels["PublicTest"]["y"]
    input["PrivateTest"]["y"] = labels["PrivateTest"]["y"]

    return input


"""
{
    "Training": {
        "x": (N, 48, 48, 1),
    },
    "PublicTest": {
        "x": (N, 48, 48, 1),
    },
    "PrivateTest": {
        "x": (N, 48, 48, 1),
    }
}
"""
def load_dataset_pixels() -> dict:
    input = {}

    with open(f"{EMOTION_DIR_PATH}/fer2013.csv") as old_fer_file:
        for row in islice(csv.reader(old_fer_file), 1, None):
            [_, pixels, usage] = row
            pixels = np.asarray(pixels.split(" "), np.uint8).reshape(48, 48, 1)

            # Normalize pixels
            pixels = pixels / 255.0

            if input.get(usage) and input[usage].get("x"):
                input[usage]["x"].append(pixels)

            else:
                input[usage] = {"x": [pixels]}

        input["Training"]["x"] = np.array(input["Training"]["x"])
        input["PublicTest"]["x"] = np.array(input["PublicTest"]["x"])
        input["PrivateTest"]["x"] = np.array(input["PrivateTest"]["x"])

    return input


"""
{
    "Training": {
        "y": (N,)
    },
    "PublicTest": {
        "y": (N,)
    },
    "PrivateTest": {
        "y": (N,)
    }
}
"""
def load_dataset_labels() -> dict:
    input = {}

    with open(f"{EMOTION_DIR_PATH}/fer2013new.csv") as new_fer_file:
        for row in islice(csv.reader(new_fer_file), 1, None):
            [
                usage,
                _,
                neutral_vote_count,
                happiness_vote_count,
                surprise_vote_count,
                sadness_vote_count,
                anger_vote_count,
                disgust_vote_count,
                fear_vote_count,
                contempt_vote_count,
                unknown_vote_count,
                nf_vote_count
            ] = row

            vote_counts = [
                neutral_vote_count,
                happiness_vote_count,
                surprise_vote_count,
                sadness_vote_count,
                anger_vote_count,
                disgust_vote_count,
                fear_vote_count,
                contempt_vote_count,
                unknown_vote_count,
                nf_vote_count
            ]

            if input.get(usage) and input[usage].get("y"):
                input[usage]["y"].append(vote_counts.index(max(vote_counts)))

            else:
                input[usage] = {"y": [vote_counts.index(max(vote_counts))]}

        input["Training"]["y"] = np.array(input["Training"]["y"])
        input["PublicTest"]["y"] = np.array(input["PublicTest"]["y"])
        input["PrivateTest"]["y"] = np.array(input["PrivateTest"]["y"])

    return input


def make_cnn(input_shape: tuple, output_shape: tuple) -> Sequential:
    model = Sequential()

    # Block-1
    model.add(
        Conv2D(
            filters=32, # Control the size of the convolution layer
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=MinMaxNorm(min_value=0.0625, max_value=4),
            input_shape=input_shape
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.5))

    # Block-2
    model.add(
        Conv2D(
            filters=64, # Control the size of the convolution layer
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=MinMaxNorm(min_value=0.0625, max_value=4),
            input_shape=input_shape
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.5))

    # Block-3
    model.add(
        Conv2D(
            filters=128, # Control the size of the convolution layer
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=MinMaxNorm(min_value=0.0625, max_value=4),
            input_shape=input_shape
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.5))

    # Block-4
    model.add(Flatten())
    model.add(
        Dense(
            units=256,
            kernel_initializer="he_normal",
            kernel_constraint=MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(Dropout(0.5))

    # Block-5
    model.add(
        Dense(
            units=128,
            kernel_initializer="he_normal",
            kernel_constraint=MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(Dropout(0.5))

    # Block-6
    model.add(
        Dense(
            units=output_shape[0],
            activation="softmax",
            kernel_initializer="glorot_normal",
        )
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    return model


if __name__ == "__main__":
    main()
