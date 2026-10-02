import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    proc_dir = os.path.join("data", "processed")
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    X_train = np.load(os.path.join(proc_dir, "X_train.npy"))
    y_train = np.load(os.path.join(proc_dir, "y_train.npy"))
    X_val = np.load(os.path.join(proc_dir, "X_val.npy"))
    y_val = np.load(os.path.join(proc_dir, "y_val.npy"))

    # ANN: Flatten -> Dense (ReLU) -> Dropout -> Dense(10, Softmax)
    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(params["dense_units"], activation="relu"),
        layers.Dropout(params["dropout_rate"]),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=params["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        verbose=1
    )

    model.save(os.path.join(model_dir, "model.h5"))
    pd.DataFrame(history.history).to_csv(os.path.join(model_dir, "history.csv"), index=False)
    print("B3: Model training complete. Artifacts saved to models/")

if __name__ == "__main__":
    main()
