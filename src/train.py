import numpy as np
from keras import models
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense, Activation, SpatialDropout2D
from keras.constraints import MinMaxNorm
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, BackupAndRestore
import sklearn.utils as sklearn_utils
import matplotlib.pyplot as plt
import os
import constants
import loader


TRAIN_EPOCH = 50


def main():
    # Load training and validation dataset
    print("Loading training and validation dataset...")
    dataset = loader.load_dataset(usages=set(["Training", "PublicTest"]))
    x_train = dataset["Training"]["x"]
    y_train = dataset["Training"]["y"]
    x_val = dataset["PublicTest"]["x"]
    y_val = dataset["PublicTest"]["y"]

    # Check the dataset shape
    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_val: ", x_val.shape)
    print("y_val: ", y_val.shape)

    model: Sequential

    # Can we start from the last model
    if os.path.isdir(constants.LAST_MODEL_DIR_PATH):
        # Resume from the last model
        print("Resuming from the last model...")
        model = models.load_model(constants.LAST_MODEL_DIR_PATH)

    else:
        # Build a CNN
        print("No last model found. Building a CNN...")
        model = make_cnn(input_shape=x_train.shape[1:], output_shape=(len(constants.LABELS),))

    # Load the best validation accuracy
    try:
        print("Loading the best validation accuracy...")

        with open(constants.BEST_VAL_ACC_FILE_PATH, "r") as best_val_acc_file:
            best_val_acc = float(best_val_acc_file.read())

        print("best_val_acc: ", best_val_acc)

    except:
        print("Failed to load the accuracy. Default best_val_acc to 0.0")
        best_val_acc = 0.0

    # Load the last model epoch
    try:
        print("Loading the last model epoch...")

        with open(constants.LAST_MODEL_EPOCH_FILE_PATH, "r") as last_model_epoch_file:
            last_epoch = int(last_model_epoch_file.read())

        print("last_epoch: ", last_epoch)

    except:
        print("Failed to load the last model epoch. Default last_epoch to 0")
        last_epoch = 0

    # Train the CNN
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=16,
        epochs=last_epoch + TRAIN_EPOCH,
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
        initial_epoch=last_epoch,
        callbacks=[
            ModelCheckpoint(
                filepath=constants.BEST_MODEL_DIR_PATH,
                monitor="val_accuracy",
                verbose=1,
                save_best_only=True,
                initial_value_threshold=best_val_acc
            ),
            BackupAndRestore(backup_dir=constants.BACKUP_DIR_PATH),
            ReduceLROnPlateau(factor=1.0 / 3.0, verbose=1)
        ]
    )

    # Save the last model epoch
    print("Save the last model epoch...")
    with open(constants.LAST_MODEL_EPOCH_FILE_PATH, "w") as last_model_epoch_file:
        last_model_epoch_file.write(str(last_epoch + TRAIN_EPOCH))

    # Save the best validation accuracy
    print("Save the best validation accuracy...")
    with open(constants.BEST_VAL_ACC_FILE_PATH, "w") as best_val_acc_file:
        best_val_acc_file.write(str(max(max(history.history['val_accuracy']), best_val_acc)))

    # Save the last model
    model.save(constants.LAST_MODEL_DIR_PATH)

    # Preparing training reports
    os.mkdir(f"{constants.REPORT_DIR_PATH}/{last_epoch + 1}-{last_epoch + TRAIN_EPOCH}")

    # Summarize history for accuracy and save the result
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f"{constants.REPORT_DIR_PATH}/{last_epoch + 1}-{last_epoch + TRAIN_EPOCH}/model-acc.png")
    plt.close()

    # Summarize history for loss and save the result
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f"{constants.REPORT_DIR_PATH}/{last_epoch + 1}-{last_epoch + TRAIN_EPOCH}/model-loss.png")
    plt.close()


def make_cnn(input_shape: tuple, output_shape: tuple) -> Sequential:
    model = Sequential()

    # Block-1
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
    model.add(
        Conv2D(
            filters=64, # Control the size of the convolution layer
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.25))

    # Block-2
    model.add(
        Conv2D(
            filters=128, # Control the size of the convolution layer
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(
        Conv2D(
            filters=128, # Control the size of the convolution layer
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.25))

    # Block-3
    model.add(
        Conv2D(
            filters=256, # Control the size of the convolution layer
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(
        Conv2D(
            filters=256, # Control the size of the convolution layer
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("elu"))
    model.add(MaxPooling2D())
    model.add(SpatialDropout2D(0.25))

    # Block-4
    model.add(Flatten())
    model.add(
        Dense(
            units=128, # Control the size of the convolution layer
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
