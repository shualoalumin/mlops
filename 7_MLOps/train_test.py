import tensorflow as tf  
import numpy as np  
print("Loading data...")  
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  
X_train = x_train.reshape(-1, 28, 28, 1) / 255.0  
y_train = tf.keras.utils.to_categorical(y_train)  
print("\nBuilding model...")  
model = tf.keras.Sequential([  
    tf.keras.layers.Conv2D(102, (3,3), activation='relu', input_shape=(28, 28, 1)),  
    tf.keras.layers.Conv2D(42, (3,3), activation='relu'),  
    tf.keras.layers.Conv2D(67, (3,3), activation='relu'),  
    tf.keras.layers.Conv2D(37, (3,3), activation='relu'),  
    tf.keras.layers.Conv2D(52, (3,3), activation='relu'),  
    tf.keras.layers.Flatten(),  
    tf.keras.layers.Dense(256, activation='relu'),  
    tf.keras.layers.Dense(10, activation='softmax')  
])  
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  
print("\nTraining for 1 epoch...")  
model.fit(X_train[:1000], y_train[:1000], epochs=1, batch_size=32, verbose=0)  
print("\nChecking weights after training:")  
for layer in model.layers:  
    weights = layer.get_weights()  
    if weights:  
        print(f"{layer.name}: {np.mean(weights[0]):.6f}")  
print("\nSaving model...")  
model.save("test_model")  
print("\nLoading saved model...")  
loaded_model = tf.keras.models.load_model("test_model")  
print("\nChecking weights after loading:")  
for layer in loaded_model.layers:  
    weights = layer.get_weights()  
    if weights:  
        print(f"{layer.name}: {np.mean(weights[0]):.6f}")  
