# No Tensorflow info messages
import os, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)

# Import required libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Create Fashion MNIST dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Scale image colors
train_images = train_images / 255
test_images = test_images / 255

# Create the model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10)
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
metrics=['accuracy'])

print('')

# Print model summary
model.summary()

# Train the model
model.fit(train_images, train_labels, epochs=10)

# Evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('Test accuracy: {}\n'.format(test_acc))

# Convert model outputs from logits to probabilities
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

# Save model
probability_model.save('model')