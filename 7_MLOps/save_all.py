import tensorflow as tf  
import numpy as np  
print("Loading original model...")  
model = tf.keras.models.load_model('best_model/1/model')  
print("\nSaving as HDF5...")  
model.save('best_model/mnist_model.h5', save_format='h5')  
print("\nSaving as Keras format...")  
model.save('best_model/mnist_model.keras')  
print("\nSaving weights...")  
model.save_weights('best_model/weights')  
