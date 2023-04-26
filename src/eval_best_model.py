import numpy as np
from keras import models
from keras.models import Sequential
import sklearn.metrics as sklearn_metrics
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import constants
import loader


def main():
    # Load test dataset
    print("Loading test dataset...")
    dataset = loader.load_dataset(usages=set(["PrivateTest"]))
    x_test = dataset["PrivateTest"]["x"]
    y_test = dataset["PrivateTest"]["y"]

    # Load the model
    print("Loading the model...")
    model: Sequential = models.load_model(constants.BEST_MODEL_DIR_PATH)

    # Evaluate the model
    metrics = model.evaluate(x=x_test, y=y_test, return_dict=True)

    # Save the model evaluation result
    with open(f"{constants.REPORT_DIR_PATH}/model-eval.txt", "w") as model_eval_file:
        model_eval_file.write(str(metrics))

    # Predict from the model
    y_pred = model.predict(x=x_test)
    y_pred = np.argmax(y_pred, axis=1)

    # Plot a confusion matrix and save the result
    result = sklearn_metrics.confusion_matrix(y_test, y_pred, normalize="pred")
    df_cm = pd.DataFrame(result, index=constants.LABELS, columns=constants.LABELS)
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(f"{constants.REPORT_DIR_PATH}/confusion-matrix.png")
    plt.close()


if __name__ == "__main__":
    main()
