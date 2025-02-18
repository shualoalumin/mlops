import tensorflow as tf  
import numpy as np  
import requests  
import json  
print("Loading test data...")  
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  
test_image = x_test[0].reshape(1, 28, 28, 1).astype('float32') / 255.0  
print(f"\nActual digit: {y_test[0]}")  
data = {"signature_name": "serving_default", "instances": test_image.tolist()}  
print("\nMaking prediction...")  
response = requests.post("http://localhost:8501/v1/models/mnist:predict", data=json.dumps(data), headers={"content-type": "application/json"})  
predictions = response.json()["predictions"][0]  
print("\nPrediction probabilities:")  
for i, prob in enumerate(predictions):  
    print(f"Digit {i}: {prob*100:.2f}%")  
print(f"\nPredicted digit: {np.argmax(predictions)}")  
