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
test_data = pd.read_csv('data/sign_mnist_test.csv')

# Convert the data to numpy arrays
train_data = train_data.to_numpy()
test_data = test_data.to_numpy()

# Randomize the data (even if it looks like it is already randomized (we never know))
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
                    batch_size=32,
                    validation_split=0.2,
                    callbacks=[early_stopping])
# Save the model
model.save('sign_language_model.keras')

### Evaluate the model
# accuracy represents the performance of the model
test_loss, test_accuracy, test_recall = model.evaluate(test_features, test_labels)
print('Test accuracy :', test_accuracy)
print('Test recall :', test_recall)

# Predict the labels for the test set
predictions = model.predict(test_features)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(test_labels, axis=1)

# Confusion matrix
conf_matrix = metrics.confusion_matrix(true_classes, predicted_classes)

# Letters for axis labels
labels = [chr(i) for i in range(65, 91)]  # 'A' to 'Z'

# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# ROC curve and AUC
fpr = {}
tpr = {}
roc_auc = {}

for i in range(26):
    fpr[i], tpr[i], _ = metrics.roc_curve(test_labels[:, i], predictions[:, i])
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

# Plot ROC curve for each class
plt.figure(figsize=(12, 10))
for i in range(26):
    plt.plot(fpr[i], tpr[i], label=f'Class {labels[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
