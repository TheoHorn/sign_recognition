import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the data
train_data = pd.read_csv('data/sign_mnist_train.csv')
test_data = pd.read_csv('data/sign_mnist_test.csv')

# Convert the data to numpy arrays
train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

# Split the data into features and labels
train_features = train_data[:, 1:]
train_labels = train_data[:, 0]

test_features = test_data[:, 1:]
test_labels = test_data[:, 0]

# Normalize the features
train_features = train_features / 255
test_features = test_features / 255

# Reshape the features for the model
train_features = train_features.reshape(-1, 28, 28, 1)
test_features = test_features.reshape(-1, 28, 28, 1)

# Convert labels to categorical
train_labels = tf.keras.utils.to_categorical(train_labels, 26)
test_labels = tf.keras.utils.to_categorical(test_labels, 26)

# Create the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(26, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_features, train_labels, epochs=10, batch_size=32)

# Save the model
model.save('sign_language_model.keras')

### Evaluate the model
# accuracy represents the performance of the model
test_loss, test_accuracy = model.evaluate(test_features, test_labels)
print('Test loss :', test_loss)
print('Test accuracy :', test_accuracy)
