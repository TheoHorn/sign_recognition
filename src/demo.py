import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import load_model

# Load the model
model = load_model('sign_language_model.keras')

# Load the test data
test_data = pd.read_csv('data/sign_mnist_test.csv')
test_data = test_data.to_numpy()
np.random.shuffle(test_data)

# Print the 10 first predictions
predictions = model.predict(test_data[:10, 1:].reshape(-1, 28, 28, 1) / 255)
for i in range(10):
    pred_indices = np.argsort(predictions[i])[::-1]  # Sort indices in descending order
    print("Prediction : ")
    print(" 1st : ", chr(pred_indices[0] + 65)," with ", predictions[i][pred_indices[0]])
    print(" 2nd : ", chr(pred_indices[1] + 65), " with ", predictions[i][pred_indices[1]])
    print(" 3rd : ", chr(pred_indices[2] + 65), " with ", predictions[i][pred_indices[2]])
    print("...")

# Show the first 10 images
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
for i in range(10):
    axs[i // 5, i % 5].imshow(test_data[i, 1:].reshape(28, 28), cmap='gray')
    axs[i // 5, i % 5].set_title(chr(test_data[i, 0] + 65))
plt.show()
