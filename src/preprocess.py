import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["preprocess"]

    raw_dir = os.path.join("data", "raw")
    proc_dir = os.path.join("data", "processed")
    os.makedirs(proc_dir, exist_ok=True)

    X_train_raw = np.load(os.path.join(raw_dir, "X_train.npy"))
    y_train_raw = np.load(os.path.join(raw_dir, "y_train.npy"))
    X_test_raw = np.load(os.path.join(raw_dir, "X_test.npy"))
    y_test = np.load(os.path.join(raw_dir, "y_test.npy"))

    # Pixel normalization to [0, 1]
    X_train_norm = X_train_raw.astype("float32") / 255.0
    X_test_norm = X_test_raw.astype("float32") / 255.0

    # Train / Validation stratified split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_norm,
        y_train_raw,
        test_size=params["val_size"],
        random_state=params["seed"],
        stratify=y_train_raw
    )

    np.save(os.path.join(proc_dir, "X_train.npy"), X_train)
    np.save(os.path.join(proc_dir, "y_train.npy"), y_train)
    np.save(os.path.join(proc_dir, "X_val.npy"), X_val)
    np.save(os.path.join(proc_dir, "y_val.npy"), y_val)
    np.save(os.path.join(proc_dir, "X_test.npy"), X_test_norm)
    np.save(os.path.join(proc_dir, "y_test.npy"), y_test)
    print(f"B2: Preprocessed data saved to {proc_dir}")

if __name__ == "__main__":
    main()
