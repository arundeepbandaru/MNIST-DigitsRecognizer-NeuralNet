import numpy as np
import pandas as pd
import tensorflow as tf
from keras.datasets import mnist
from keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image
from skimage.transform import resize

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train_normalized = x_train / 255.0
x_test_normalized = x_test / 255.0

# Function to plot training curves
def plot_curve(epochs, hist, list_of_metrics):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    for m in list_of_metrics:
        x = hist[m]
        plt.plot(epochs[1:], x[1:], label=m)
    plt.legend()

# Create the model architecture
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(units=256, activation='relu'))
    model.add(tf.keras.layers.Dense(units=128, activation='relu'))
    model.add(tf.keras.layers.Dropout(rate=0.1))
    model.add(tf.keras.layers.Dense(units=10, activation='softmax'))
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])
    return model

# Train the model
def train_model(model, train_features, train_labels, epochs, batch_size, validation_split):
    history = model.fit(x=train_features, y=train_labels, batch_size=batch_size, epochs=epochs, 
                        shuffle=True, validation_split=validation_split)
    epochs = history.epoch
    hist = pd.DataFrame(history.history)
    return epochs, hist

# Hyperparameters
epochs = 25
batch_size = 1000
validation_split = 0

# Create and train the model
my_model = create_model()
epochs, hist = train_model(my_model, x_train_normalized, y_train, epochs, batch_size, validation_split)

# Plot training curves
list_of_metrics_to_plot = ['accuracy']
plot_curve(epochs, hist, list_of_metrics_to_plot)

# Evaluate the model on the test set
my_model.evaluate(x_test_normalized, y_test, batch_size)

# Create a confusion matrix for the test set
y_pred = my_model.predict(x_test_normalized)
y_pred_labels = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred_labels, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
sns.heatmap(cm, annot=True, fmt='g', xticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], yticklabels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.ylabel('Actual', fontsize=14)
plt.xlabel('Prediction', fontsize=14)
plt.title('Confusion Matrix', fontsize=20)
plt.show()
