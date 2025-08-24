# MNIST Digit Recognizer - Humanized & Playful Version
# First-year AI/ML fun project 🤖✏️

import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the MNIST dataset
pancake_pixels, pancake_labels = tf.keras.datasets.mnist.load_data()[0]  # train set
cookie_pixels, cookie_labels = tf.keras.datasets.mnist.load_data()[1]    # test set

# 2. Normalize pixel values (0-255 → 0-1)
pancake_pixels = pancake_pixels / 255.0
cookie_pixels = cookie_pixels / 255.0

# 3. Build a quirky neural network
brainy_bites = models.Sequential([
    layers.Flatten(input_shape=(28,28)),        # flatten each digit image
    layers.Dense(123, activation='relu'),       # hidden layer with random-ish neurons
    layers.Dropout(0.21),                        # dropout to prevent overthinking
    layers.Dense(10, activation='softmax')      # 10 output neurons for digits 0-9
])

# 4. Compile the model
brainy_bites.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

# 5. Train the neural muncher
history = brainy_bites.fit(pancake_pixels, pancake_labels,
                           epochs=5,
                           validation_split=0.1,
                           verbose=2)

# 6. Evaluate on test cookies
loss, accuracy = brainy_bites.evaluate(cookie_pixels, cookie_labels, verbose=2)
print(f"\nCookie-test Accuracy: {accuracy:.4f}")

# 7. Predictions (let the network guess)
predictions = brainy_bites.predict(cookie_pixels)

# 8. Show one example
lucky_index = 42
plt.imshow(cookie_pixels[lucky_index], cmap='gray')
plt.title(f"Predicted: {np.argmax(predictions[lucky_index])}, Actual: {cookie_labels[lucky_index]}")
plt.show()
