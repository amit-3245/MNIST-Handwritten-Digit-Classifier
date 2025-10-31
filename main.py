
#  Image Classification using Neural Networks (MNIST)


# Step 1️: Import Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


# Step 2️: Load a Prebuilt Dataset (MNIST Handwritten Digits)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print("Training Data Shape:", x_train.shape)
print("Test Data Shape:", x_test.shape)


# Step 3️: Data Preprocessing

# Normalize pixel values (0–255 → 0–1 range)
x_train = x_train / 255.0
x_test = x_test / 255.0


# Step 4️: Build the Neural Network Model

model = tf.keras.models.Sequential([
    # Flatten Layer: Converts each 28x28 image into a 1D vector (784 pixels)
    tf.keras.layers.Flatten(input_shape=(28, 28)),

    # Hidden Layer: 128 neurons with ReLU activation for non-linearity
    tf.keras.layers.Dense(128, activation='relu'),

    # Dropout Layer: Prevents overfitting by randomly turning off 20% of neurons
    tf.keras.layers.Dropout(0.2),

    # Output Layer: 10 neurons for 10 classes (digits 0–9)
    tf.keras.layers.Dense(10)
])

1


# Step 5️: Compile the Model-

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Step 6️: Train the Model

# Note: 'epochs' not 'epoch'
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))


# Step 7️: Evaluate the Model

test_loss, test_acc = model.evaluate(x_test, y_test)
print("\n Test Accuracy:", round(test_acc * 100, 2), "%")


# Step 8️: Make Predictions

# Apply softmax to convert logits → probabilities
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)

# Example: predicted label for the first image
predicted_label = np.argmax(predictions[0])
print("\nPredicted Label:", predicted_label)
print("Actual Label:", y_test[0])


# Step 9️: Visualize Predictions

plt.figure(figsize=(4,4))
plt.imshow(x_test[0], cmap='gray')
plt.title(f" Predicted: {predicted_label} | Actual: {y_test[0]}")
plt.axis('off')
plt.show()


# Step 10: Visualize Accuracy & Loss Graphs

plt.figure(figsize=(10,4))

# Accuracy Graph
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title(' Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Graph
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title(' Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()
