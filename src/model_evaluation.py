import pandas as pd
import numpy as np
import tensorflow as tf
from keras import layers, models
from matplotlib import pyplot as plt
import sklearn.metrics as metrics
import seaborn as sns

# Load the data
test_data = pd.read_csv('data/sign_mnist_test.csv')
test_data = test_data.to_numpy()

test_features = test_data[:, 1:]
test_labels = test_data[:, 0]

test_features = test_features / 255

test_features = test_features.reshape(-1, 28, 28, 1)

test_labels = tf.keras.utils.to_categorical(test_labels, 26)

# Load the model
model = models.load_model('sign_language_model.keras')

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
labels = [chr(i) for i in range(65, 91) ]  # 'A' to 'Z' 
labels_jz = [chr(i) for i in range(65, 91) if i != 74 and i != 90]  # 'A' to 'Z' excluding 'J' and 'Z'


# Plot confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels_jz, yticklabels=labels_jz)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('images/confusion_matrix.png')

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
plt.savefig('images/roc_curve.png')
