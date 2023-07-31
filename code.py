import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the deep learning model
def create_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

# Load and preprocess the dataset
def load_dataset():
    # You can replace this with your own dataset loading and preprocessing code
    # Make sure to convert your dataset into feature and target variables (X and y)
    # X should be a 2D array-like object with shape (num_samples, num_features)
    # y should be a 1D array-like object with shape (num_samples,)
    dataset = pd.read_csv('dataset.csv')
    X = dataset[['k__Archaea']].values
    y = dataset[['disease']].values
    return X, y

# Train the model
def train_model(model, X, y):
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)

# Evaluate the model
def evaluate_model(model, X, y):
    loss = model.evaluate(X, y, verbose=0)
    print("Mean Squared Error:", loss)

# Plot the predictions
def plot_predictions(model, X, y):
    y_pred = model.predict(X)
    plt.scatter(X, y, label='Actual')
    plt.scatter(X, y_pred, color='r', label='Predicted')
    plt.legend()
    plt.xlabel('Feature Variable')
    plt.ylabel('Target Variable')
    plt.title('Predicted vs Actual')
    plt.show()

# Main code
def main():
    # Load the dataset
    X, y = load_dataset()

    # Create the model
    model = create_model(input_shape=(X.shape[1],))

    # Train the model
    train_model(model, X, y)

    # Evaluate the model
    evaluate_model(model, X, y)

    # Plot the predictions
    plot_predictions(model, X, y)

if __name__ == '__main__':
    main()
