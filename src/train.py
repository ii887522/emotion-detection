import numpy as np
import tensorflow as tf
import sklearn.utils as sklearn_utils
import matplotlib.pyplot as plt
import os
import constants
import loader
import csv
from itertools import islice


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

    model: tf.keras.Sequential

    # Can we start from the last model
    if os.path.isdir(constants.LAST_MODEL_DIR_PATH):
        # Resume from the last model
        print("Resuming from the last model...")
        model = tf.keras.models.load_model(constants.LAST_MODEL_DIR_PATH)

    else:
        # Build a CNN
        print("No last model found. Building a CNN...")
        model = make_cnn(input_shape=x_train.shape[1:], output_shape=(len(constants.LABELS),))

    best_val_acc = 0.0
    last_epoch = -1

    # Load the best validation accuracy and last epoch
    try:
        print("Loading the best validation accuracy...")
        print("Loading the last epoch...")

        with open(constants.TRAIN_LOG_FILE_PATH) as train_log_file:
            for row in islice(csv.reader(train_log_file), 1, None):
                [epoch, _, _, _, val_acc, _] = row
                best_val_acc = max(best_val_acc, float(val_acc))
                last_epoch = int(epoch)

            print("best_val_acc", best_val_acc)
            print("last_epoch", last_epoch)

    except:
        print("Failed to load the best validation accuracy. Default best_val_acc to 0.0")
        print("Failed to load the last epoch. Default last_epoch to -1")

    to_epoch = ((last_epoch + 1) // constants.TRAIN_EPOCH + 1) * constants.TRAIN_EPOCH

    # Train the CNN
    model.fit(
        x=x_train,
        y=y_train,
        batch_size=constants.BATCH_SIZE,
        epochs=to_epoch,
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
        initial_epoch=last_epoch + 1,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=constants.BEST_MODEL_DIR_PATH,
                monitor="val_accuracy",
                verbose=1,
                save_best_only=True,
                initial_value_threshold=best_val_acc
            ),
            tf.keras.callbacks.BackupAndRestore(backup_dir=constants.BACKUP_DIR_PATH),
            tf.keras.callbacks.ReduceLROnPlateau(factor=1.0 / 3.0, patience=7, verbose=1),
            tf.keras.callbacks.CSVLogger(filename=constants.TRAIN_LOG_FILE_PATH, append=True)
        ]
    )

    # Save the last model
    print("Saving the last model...")
    model.save(constants.LAST_MODEL_DIR_PATH)

    # Preparing training reports
    print("Generating training reports...")
    os.mkdir(f"{constants.REPORT_DIR_PATH}/{to_epoch - constants.TRAIN_EPOCH + 1}-{to_epoch}")

    with open(constants.TRAIN_LOG_FILE_PATH) as train_log_file:
        epoches = []
        accs = []
        val_accs = []
        losses = []
        val_losses = []

        for row in islice(csv.reader(train_log_file), to_epoch - constants.TRAIN_EPOCH + 1, None):
            [epoch, acc, loss, _, val_acc, val_loss] = row
            epoches.append(int(epoch) + 1)
            accs.append(float(acc))
            val_accs.append(float(val_acc))
            losses.append(float(loss))
            val_losses.append(float(val_loss))

        # Summarize history for loss and save the result
        print("Generating model loss report...")
        plt.plot(epoches, losses)
        plt.plot(epoches, val_losses)
        plt.title('Model Loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f"{constants.REPORT_DIR_PATH}/{to_epoch - constants.TRAIN_EPOCH + 1}-{to_epoch}/model-loss.png")
        plt.close()

        # Summarize history for accuracy and save the result
        print("Generating model accuracy report...")
        plt.plot(epoches, accs)
        plt.plot(epoches, val_accs)
        plt.title('Model Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.savefig(f"{constants.REPORT_DIR_PATH}/{to_epoch - constants.TRAIN_EPOCH + 1}-{to_epoch}/model-acc.png")
        plt.close()


def make_cnn(input_shape: tuple, output_shape: tuple) -> tf.keras.Sequential:
    model = tf.keras.Sequential()

    # Convolution block #1
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, # Control the size of the convolution layer
            activation="elu",
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4, axis=[0, 1, 2, 3]),
            bias_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4),
            input_shape=input_shape
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, # Control the size of the convolution layer
            activation="elu",
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4, axis=[0, 1, 2, 3]),
            bias_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))

    # Convolution block #2
    model.add(
        tf.keras.layers.Conv2D(
            filters=128, # Control the size of the convolution layer
            activation="elu",
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4, axis=[0, 1, 2, 3]),
            bias_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(
        tf.keras.layers.Conv2D(
            filters=128, # Control the size of the convolution layer
            activation="elu",
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4, axis=[0, 1, 2, 3]),
            bias_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))

    # Convolution block #3
    model.add(
        tf.keras.layers.Conv2D(
            filters=256, # Control the size of the convolution layer
            activation="elu",
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4, axis=[0, 1, 2, 3]),
            bias_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(
        tf.keras.layers.Conv2D(
            filters=256, # Control the size of the convolution layer
            activation="elu",
            kernel_size=3,
            kernel_initializer="he_normal",
            kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4, axis=[0, 1, 2, 3]),
            bias_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D())
    model.add(tf.keras.layers.Dropout(0.2))

    # Dense block #1
    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(
            units=512, # Control the size of the convolution layer
            activation="elu",
            kernel_initializer="he_normal",
            kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4, axis=[0, 1]),
            bias_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    # Dense block #2
    model.add(
        tf.keras.layers.Dense(
            units=256, # Control the size of the convolution layer
            activation="elu",
            kernel_initializer="he_normal",
            kernel_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4, axis=[0, 1]),
            bias_constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0625, max_value=4),
        )
    )
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    # Dense block #3
    model.add(
        tf.keras.layers.Dense(
            units=output_shape[0],
            activation="softmax",
            kernel_initializer="glorot_normal",
        )
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001 / 3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


if __name__ == "__main__":
    main()
