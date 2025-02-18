import tensorflow as tf
import numpy as np

# Load and preprocess a small amount of data
print("Loading data...")
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
X_train = x_train[:1000].reshape(-1, 28, 28, 1) / 255.0
y_train = tf.keras.utils.to_categorical(y_train[:1000])

# Build a smaller model
print("\nBuilding model...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train for one epoch
print("\nTraining...")
model.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1)

# Check weights before saving
print("\nWeights before saving:")
for layer in model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"{layer.name}: {np.mean(weights[0]):.6f}")

# Save and reload
print("\nSaving and reloading model...")
model.save("test_model")
loaded_model = tf.keras.models.load_model("test_model")

# Check weights after loading
print("\nWeights after loading:")
for layer in loaded_model.layers:
    weights = layer.get_weights()
    if weights:
        print(f"{layer.name}: {np.mean(weights[0]):.6f}") 