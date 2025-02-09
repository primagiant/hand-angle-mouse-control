import numpy as np
import tensorflow_decision_forests as tfdf
from tensorflow import keras

class HandGestureRandomForestClassifier:
    def __init__(self, model_path="model/random_forest_model"):
        # Load the trained Random Forest model
        self.model = keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")

    def __call__(self, landmark_list):
        # Convert landmarks to a NumPy array & ensure float32 type
        input_data = np.array(landmark_list, dtype=np.float32).reshape(1, -1)

        # Make a prediction
        prediction = self.model(input_data)

        # Extract the tensor using the correct key "output_1"
        y_new_pred = prediction["output_1"].numpy()  # Convert to NumPy array

        # Use argmax to get the predicted class
        predicted_class = np.argmax(y_new_pred, axis=1)

        return predicted_class
