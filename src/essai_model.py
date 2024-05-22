import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models

# Load the data
train_data = pd.read_csv('data/sign_mnist_train.csv')
test_data = pd.read_csv('data/sign_mnist_test.csv')

# Convert the data to numpy arrays
train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

# Randomize the data
np.random.shuffle(train_data)
np.random.shuffle(test_data)

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

# Function to calculate the F1-score
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=1), tf.float32)
    
    tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32), axis=0)
    fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32), axis=0)
    fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32), axis=0)
    
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
    return tf.reduce_mean(f1)

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
              metrics=['accuracy', 'recall', f1_score])

# Train the model
model.fit(train_features, train_labels, epochs=3, batch_size=8)

# Save the model
model.save('sign_language_model.keras')

### Evaluate the model
# accuracy represents the performance of the model
test_loss, test_accuracy, test_recall, test_f1 = model.evaluate(test_features, test_labels)
print('Test accuracy :', test_accuracy)
print('Test recall :', test_recall)
print('Test F1-score :', test_f1)
