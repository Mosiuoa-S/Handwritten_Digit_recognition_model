# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import time
import os

def load_and_preprocess_data():
    """Load and preprocess the MNIST dataset."""
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape the data: [samples][width][height][channels]
    # For grayscale images, we have 1 channel
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # Convert from uint8 to float32 and normalize to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to one-hot encoding
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    return (x_train, y_train), (x_test, y_test)

def create_cnn_model():
    """Create a Convolutional Neural Network model."""
    model = Sequential()
    
    # First convolutional layer
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    
    # Second convolutional layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    
    # Max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Dropout to prevent overfitting
    model.add(Dropout(0.25))
    
    # Flatten layer to convert 2D features to 1D
    model.add(Flatten())
    
    # Fully connected layer
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    # Output layer with 10 neurons (one for each digit)
    model.add(Dense(10, activation='softmax'))
    
    # Compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    model.summary()
    return model

def train_model(model, x_train, y_train, x_test, y_test, batch_size=128, epochs=10):
    """Train the model and return the history."""
    start_time = time.time()
    
    # Train the model
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test)
    )
    
    end_time = time.time()
    print(f"Training time: {end_time - start_time:.2f} seconds")
    
    return history

def evaluate_model(model, x_test, y_test):
    """Evaluate the model and print the results."""
    # Evaluate the model
    score = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {score[0]:.4f}")
    print(f"Test accuracy: {score[1]:.4f}")
    
    return score

def plot_training_history(history):
    """Plot the training and validation accuracy and loss."""
    # Plot accuracy
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def visualize_predictions(model, x_test, y_test):
    """Visualize some predictions on the test set."""
    # Predict classes for test images
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Select random images from test set
    num_samples = 10
    random_indices = np.random.choice(x_test.shape[0], num_samples, replace=False)
    
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(random_indices):
        plt.subplot(2, 5, i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        predicted_label = y_pred_classes[idx]
        true_label = y_true_classes[idx]
        
        if predicted_label == true_label:
            color = 'green'
        else:
            color = 'red'
            
        plt.title(f"Pred: {predicted_label}, True: {true_label}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('predictions.png')
    plt.show()

def save_model(model, filename='mnist_model.h5'):
    """Save the trained model."""
    model.save(filename)
    print(f"Model saved as {filename}")

def load_model(filename='mnist_model.h5'):
    """Load a trained model."""
    if os.path.exists(filename):
        model = tf.keras.models.load_model(filename)
        print(f"Model loaded from {filename}")
        return model
    else:
        print(f"Model file {filename} not found.")
        return None

def main():
    print("MNIST Handwritten Digit Recognition")
    print("-" * 40)
    
    # Check for model file
    model_filename = 'mnist_model.h5'
    model = load_model(model_filename)
    
    if model is None:
        print("Training new model...")
        # Load and preprocess data
        (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
        
        # Create model
        model = create_cnn_model()
        
        # Train model
        history = train_model(model, x_train, y_train, x_test, y_test)
        
        # Plot training history
        plot_training_history(history)
        
        # Save model
        save_model(model, model_filename)
    else:
        # Load data for evaluation
        (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Evaluate model
    evaluate_model(model, x_test, y_test)
    
    # Visualize predictions
    visualize_predictions(model, x_test, y_test)
    
    print("\nTry your own digit:")
    print("You can use the function predict_digit(model, image_path) to predict your own images.")


def predict_own_digit(model, image_path):
    """Predict a digit from an image file."""
    from PIL import Image
    import numpy as np

    
    # Load image
    img = Image.open(image_path).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to MNIST format
    
    # Convert to numpy array
    img_array = np.array(img)
    img_array = 255 - img_array  # Invert colors if needed (MNIST has white digits on black)
    
    # Normalize and reshape
    img_array = img_array.astype('float32') / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)
    
    
    print("The predicted didgit is",predicted_digit)
    return None 

if __name__ == "__main__":
    main()