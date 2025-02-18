import tensorflow as tf  
import numpy as np  
print("Loading MNIST data...")  
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  
test_image = x_test[0].reshape(1, 28, 28, 1) / 255.0  
print(f"\nActual digit: {y_test[0]}")  
print("\nLoading model...")  
model = tf.keras.models.load_model('best_model/1/model')  
print("\nMaking predictions...")  
test_pred = model.predict(test_image)  
print("\nPrediction results:")  
for i, prob in enumerate(test_pred[0]):  
    print(f"Digit {i}: {prob*100:.2f}%%") 
