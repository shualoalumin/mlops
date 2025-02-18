import json
import requests

# Create a simple test image (all zeros)
test_image = [[[0.0] * 3] * 32] * 32
test_data = [test_image]  # Batch size 1

# Prepare the data for the request
data = json.dumps({
    "signature_name": "serving_default",
    "instances": test_data
})

# Send request to TensorFlow Serving
headers = {"content-type": "application/json"}
url = "http://localhost:8501/v1/models/cifar10_model:predict"
response = requests.post(url, data=data, headers=headers)

if response.status_code == 200:
    predictions = response.json()['predictions']
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Get the class with highest probability
    max_prob_index = predictions[0].index(max(predictions[0]))
    print(f"Predicted class: {class_names[max_prob_index]} (class {max_prob_index})")
    print(f"Prediction probabilities: {predictions[0]}")
else:
    print("Error:", response.text) 