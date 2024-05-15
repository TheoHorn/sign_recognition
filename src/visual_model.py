import keras
from keras.utils import plot_model

# Load your model
model = keras.models.load_model('sign_language_model.keras')

# Plot the model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
