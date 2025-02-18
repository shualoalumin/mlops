import tensorflow as tf  
import numpy as np  
def print_predictions(name, predictions):  
    print(f"\n{name} predictions:")  
    for i, prob in enumerate(predictions[0]):  
        print(f"Digit {i}: {prob*100:.2f}%%")  
    print(f"Predicted digit: {np.argmax(predictions[0])}")  
print("Loading test data...")  
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  
test_image = x_test[0].reshape(1, 28, 28, 1).astype('float32') / 255.0  
print(f"\nActual digit: {y_test[0]}")  
print("\nTesting H5 model...")  
h5_model = tf.keras.models.load_model('best_model/mnist_model.h5')  
print_predictions("H5 model", h5_model.predict(test_image))  
print("\nTesting Keras model...")  
keras_model = tf.keras.models.load_model('best_model/mnist_model.keras')  
print_predictions("Keras model", keras_model.predict(test_image))  
print("\nTesting with weights...")  
model = tf.keras.models.load_model('best_model/1/model')  
model.load_weights('best_model/weights')  
print_predictions("Weights model", model.predict(test_image))  
