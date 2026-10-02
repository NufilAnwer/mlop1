import os
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

def main():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["preprocess"]
        
    x_train_full = np.load('data/raw/x_train.npy')
    y_train_full = np.load('data/raw/y_train.npy')
    x_test = np.load('data/raw/x_test.npy')
    y_test = np.load('data/raw/y_test.npy')
    
    x_train_full = x_train_full / 255.0
    x_test = x_test / 255.0
    
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_full, y_train_full, 
        test_size=params["test_size"], 
        random_state=params["seed"]
    )
    
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/x_train.npy', x_train)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/x_val.npy', x_val)
    np.save('data/processed/y_val.npy', y_val)
    np.save('data/processed/x_test.npy', x_test)
    np.save('data/processed/y_test.npy', y_test)
    print("Processed data saved.")

if __name__ == '__main__':
    main()
