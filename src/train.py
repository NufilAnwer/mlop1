import os
import yaml
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]
        
    x_train = np.load('data/processed/x_train.npy')
    y_train = np.load('data/processed/y_train.npy')
    x_val = np.load('data/processed/x_val.npy')
    y_val = np.load('data/processed/y_val.npy')
    
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(params["dense_units"], activation='relu'),
        Dropout(params["dropout_rate"]),
        Dense(10, activation='softmax')
    ])
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
                  
    history = model.fit(x_train, y_train,
                        epochs=params["epochs"],
                        batch_size=params["batch_size"],
                        validation_data=(x_val, y_val))
                        
    os.makedirs('models', exist_ok=True)
    model.save('models/model.h5')
    
    hist_df = pd.DataFrame(history.history)
    hist_df.to_csv('models/history.csv', index=False)
    print("Model and history saved.")

if __name__ == '__main__':
    main()
