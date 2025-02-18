import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('cifar10_tuned_model')

# Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model('cifar10_tuned_model')
tflite_model = converter.convert()

# Save the TFLite model
with open('cifar10_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Get model signature
interpreter = tf.lite.Interpreter(model_path='cifar10_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nModel Signature Details:")
print("Input:", input_details)
print("\nOutput:", output_details) 