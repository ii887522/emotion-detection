# Author: Amos Teo Hua An

import numpy as np
import TLConstants
import os
import TLLoader
import csv
import tensorflow as tf
import sklearn.utils as sklearn_utils
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from itertools import islice

new_model: tf.keras.Sequential

# Can we start from checkpoint ?
if os.path.isfile(TLConstants.LAST_TL_MODEL_DIR_PATH):
    # Resume from the last checkpoint
    new_model = tf.keras.models.load_model(TLConstants.LAST_TL_MODEL_DIR_PATH)

else:

    ## pretrained model
    model = tf.keras.applications.MobileNetV2(input_shape=(48,48,3), include_top=False)

    ## transfer learning - tune the weights and start from last check point
    base_input = model.layers[0].input
    final_output = model.layers[-1].output
    final_output = layers.GlobalMaxPooling2D()(final_output)

    final_output = layers.Dense(128)(final_output) ## adding a new layer, after output of global pooling layer
    final_output = layers.Activation('relu')(final_output) ## activate the function
    final_output = layers.Dense(64)(final_output)
    final_output = layers.Activation('relu')(final_output)
    final_output = layers.Dense(10,activation='softmax')(final_output) ##since we only have 10 classes we put 10

    ## create new model
    new_model = tf.keras.Model(inputs = base_input, outputs=final_output)
    new_model.compile(loss="sparse_categorical_crossentropy", optimizer = tf.keras.optimizers.Adam(0.00001), metrics = ["accuracy"],  run_eagerly=True)
    new_model.summary()

best_val_acc = 0.0
last_epoch = -1

# Load the best validation accuracy and last epoch
try:
    print("Loading the best validation accuracy...")
    print("Loading the last epoch...")

    with open(TLConstants.TL_TRAIN_LOG_FILE_PATH) as train_log_file:
        for row in islice(csv.reader(train_log_file), 1, None):
            [epoch, _, _, _, val_acc, _] = row
            best_val_acc = max(best_val_acc, float(val_acc))
            last_epoch = int(epoch)

        print("best_val_acc", best_val_acc)
        print("last_epoch", last_epoch)

except:
    print("Failed to load the best validation accuracy. Default best_val_acc to 0.0")
    print("Failed to load the last epoch. Default last_epoch to -1")

to_epoch = ((last_epoch + 1) // TLConstants.TRAIN_EPOCH + 1) * TLConstants.TRAIN_EPOCH

X = TLLoader.load_dataset(set(["Training", "PublicTest", "PrivateTest"]))

## training time
new_model.fit(X["Training"]["x"],  X["Training"]["y"], epochs = to_epoch, validation_data=(X["PublicTest"]["x"], X["PublicTest"]["y"]),
            initial_epoch=last_epoch + 1, callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
            filepath=TLConstants.BEST_TL_MODEL_DIR_PATH,
            monitor="val_accuracy",
            verbose=1,
            save_best_only=True,
            initial_value_threshold=best_val_acc
        ),
        tf.keras.callbacks.BackupAndRestore(backup_dir=TLConstants.BACKUP_DIR_PATH),
        tf.keras.callbacks.ReduceLROnPlateau(factor=1.0 / 3.0, patience=3, verbose=1),
        tf.keras.callbacks.CSVLogger(filename=TLConstants.TL_TRAIN_LOG_FILE_PATH, append=True)
    ])
print("Saving the last model...")
new_model.save(TLConstants.LAST_TL_MODEL_DIR_PATH)
