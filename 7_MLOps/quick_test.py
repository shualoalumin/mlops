import tensorflow as tf  
import numpy as np  
print("Loading data...")  
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  
test_image = x_test[0].reshape(1, 28, 28, 1).astype('float32') / 255.0  
print(f"\nActual digit: {y_test[0]}")  
print("\nLoading original model...")  
model = tf.keras.models.load_model('best_model/1/model')  
print("\nChecking model weights...")  
for layer in model.layers:  
    weights = layer.get_weights()  
    if weights:  
        print(f"{layer.name}: {np.mean(weights[0]):.6f}")  
