import os
import numpy as np
import tensorflow as tf

def main():
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    np.save(os.path.join(raw_dir, "X_train.npy"), X_train)
    np.save(os.path.join(raw_dir, "y_train.npy"), y_train)
    np.save(os.path.join(raw_dir, "X_test.npy"), X_test)
    np.save(os.path.join(raw_dir, "y_test.npy"), y_test)
    print("B1: Raw Fashion-MNIST arrays successfully saved to data/raw/")

if __name__ == "__main__":
    main()
