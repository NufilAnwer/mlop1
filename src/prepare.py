import os
import numpy as np
import tensorflow as tf

def main():
    os.makedirs('data/raw', exist_ok=True)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    np.save('data/raw/x_train.npy', x_train)
    np.save('data/raw/y_train.npy', y_train)
    np.save('data/raw/x_test.npy', x_test)
    np.save('data/raw/y_test.npy', y_test)
    print("Raw data saved.")

if __name__ == '__main__':
    main()
