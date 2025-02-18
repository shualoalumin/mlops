import tensorflow as tf  
import numpy as np  
print("Loading data...")  
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()  
X_train = x_train.reshape(-1, 28, 28, 1)  
X_test = x_test.reshape(-1, 28, 28, 1)  
y_train = tf.keras.utils.to_categorical(y_train)  
y_test = tf.keras.utils.to_categorical(y_test)  
print("\nBuilding and training model...")  
def build_model():  
    model = tf.keras.Sequential()  
    model.add(tf.keras.Input(shape=(28, 28, 1)))  
    model.add(tf.keras.layers.Conv2D(102, (3,3), activation='relu'))  
    model.add(tf.keras.layers.Conv2D(42, (3,3), activation='relu'))  
    model.add(tf.keras.layers.Conv2D(67, (3,3), activation='relu'))  
    model.add(tf.keras.layers.Conv2D(37, (3,3), activation='relu'))  
    model.add(tf.keras.layers.Conv2D(52, (3,3), activation='relu'))  
    model.add(tf.keras.layers.Flatten())  
    model.add(tf.keras.layers.Dense(256, activation='relu'))  
    model.add(tf.keras.layers.Dense(10, activation='softmax'))  
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  
    return model  
model = build_model()  
print("\nModel summary:")  
model.summary()  
history = model.fit(X_train, y_train, epochs=5, batch_size=32)  
print("\nTesting sample prediction before saving:")  
test_image = X_test[0:1]  
print(f"Actual digit: {np.argmax(y_test[0])}")  
pred = model.predict(test_image)  
print("\nPrediction probabilities:")  
for i, p in enumerate(pred[0]):  
    print(f"Digit {i}: {p*100:.2f}%%")  
print(f"\nPredicted digit: {np.argmax(pred[0])}")  
print("\nSaving model in different formats...")  
model.save("fresh_model.h5", save_format="h5")  
model.save("fresh_model.keras")  
model.save_weights("fresh_weights")  
print("\nTesting predictions after saving:")  
for format in ["h5", "keras", "weights"]:  
    print(f"\nTesting {format} format:")  
    if format == "weights":  
        new_model = build_model()  
        new_model.load_weights("fresh_weights")  
    else:  
        new_model = tf.keras.models.load_model(f"fresh_model.{format}")  
    pred = new_model.predict(test_image)  
    print(f"Predicted digit: {np.argmax(pred[0])}")  
    for i, p in enumerate(pred[0]):  
        print(f"Digit {i}: {p*100:.2f}%%")  
