import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns

# Load the data
train_data = pd.read_csv('data/sign_mnist_train.csv')

# Convert the data to numpy arrays
train_data = train_data.to_numpy()

# Randomize the data (even if it looks like it is already randomized (we never know))
np.random.shuffle(train_data)


# Split the data into features and labels
train_features = train_data[:, 1:]
train_labels = train_data[:, 0]

# Normalize the features
train_features = train_features / 255

# Reshape the features for the model
train_features = train_features.reshape(-1, 28, 28, 1)

# Convert labels to categorical
train_labels = tf.keras.utils.to_categorical(train_labels, 26)

# Create the model
model = models.Sequential([
    layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)),
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
              metrics=['accuracy', 'recall'])

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(train_features, train_labels,
                    epochs=15,
                    batch_size=8,
                    validation_split=0.2,
                    callbacks=[early_stopping])
# Save the model
model.save('sign_language_model.keras')