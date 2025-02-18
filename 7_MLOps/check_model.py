import tensorflow as tf  
import numpy as np  
print("Loading data...")  
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  
test_image = x_test[0].reshape(1, 28, 28, 1).astype('float32') / 255.0  
print(f"\nActual digit: {y_test[0]}")  
print("\nLoading model...")  
model = tf.keras.models.load_model('best_model/1/model')  
print("\nMaking prediction...")  
predictions = model.predict(test_image)  
print("\nPrediction probabilities:")  
for i, prob in enumerate(predictions[0]):  
    print(f"Digit {i}: {prob*100:.2f}%")  
print(f"\nPredicted digit: {np.argmax(predictions[0])}")  
