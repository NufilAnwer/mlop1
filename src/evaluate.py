import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

def main():
    proc_dir = os.path.join("data", "processed")
    model_path = os.path.join("models", "model.h5")

    X_test = np.load(os.path.join(proc_dir, "X_test.npy"))
    y_test = np.load(os.path.join(proc_dir, "y_test.npy"))

    model = tf.keras.models.load_model(model_path)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

    y_pred = np.argmax(model.predict(X_test), axis=1)
    cm = confusion_matrix(y_test, y_pred)
    
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    plt.close()

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy)
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"B4: Evaluation complete. Metrics: {metrics}")

if __name__ == "__main__":
    main()
