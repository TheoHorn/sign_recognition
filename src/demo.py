import tensorflow as tf
import pandas as pd
import numpy as np
from keras.models import load_model
import argparse

parser = argparse.ArgumentParser(description='Predict sign language from images.')
parser.add_argument('--model', type=str, default='sign_language_model.keras',
                    help='Path to the model file (default: "sign_language_model.keras")')
parser.add_argument('--data', type=str, default='data/sign_mnist_test.csv',
                    help='Path to the test data file (default: "data/sign_mnist_test.csv")')

args = parser.parse_args()

# Load the model
model = load_model(args.model)

# Load the test data
test_data = pd.read_csv(args.data)
test_data = test_data.to_numpy()
np.random.shuffle(test_data)

# Print the 10 first predictions
predictions = model.predict(test_data[:10, 1:].reshape(-1, 28, 28, 1) / 255)
pred_indices = np.argsort(predictions, axis=1)[:, ::-1]  # Sort indices in descending order

df_predictions = pd.DataFrame(np.column_stack((
    np.arange(10),
    [chr(pred_indices[i, 0] + 65) for i in range(10)],
    [predictions[i, pred_indices[i, 0]] for i in range(10)],
    [chr(pred_indices[i, 1] + 65) for i in range(10)],
    [predictions[i, pred_indices[i, 1]] for i in range(10)],
    [chr(pred_indices[i, 2] + 65) for i in range(10)],
    [predictions[i, pred_indices[i, 2]] for i in range(10)]
)), columns=['ID', '1st', '1st Prob', '2nd', '2nd Prob', '3rd', '3rd Prob'])

print(df_predictions.to_string(index=False))

# Show the first 10 images
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 5, figsize=(20, 10))
for i in range(10):
    axs[i // 5, i % 5].imshow(test_data[i, 1:].reshape(28, 28), cmap='gray')
    axs[i // 5, i % 5].set_title(chr(test_data[i, 0] + 65))
plt.show()
