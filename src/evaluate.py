import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def main():
    x_test = np.load('data/processed/x_test.npy')
    y_test = np.load('data/processed/y_test.npy')
    
    model = tf.keras.models.load_model('models/model.h5')
    
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    metrics = {
        "loss": float(loss),
        "accuracy": float(accuracy)
    }
    
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10,8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('metrics_confusion_matrix.png')
    
    print("Metrics evaluated and saved.")

if __name__ == '__main__':
    main()
