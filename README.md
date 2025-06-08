# MNIST Handwritten Digit Recognition by 202101671 (Hashatsi KF), 202100003 (Sello MF), 202100402 (Sello RJ)

This program lets you train and test a neural network that can recognize handwritten digits (0-9). It uses the famous MNIST dataset as a training resource.

## Features
- Train a hanwritten digit recognition model using the MNIST dataset 
- Save and reuse the model
- Predicts a digit from an image of any file type using saved model 

## Requirements
- Make sure you have **Python 3**, **venv** and **pip** installed
    '''bash
    - sudo apt install python3 python3-venv python3-pip

- Create a virtual environment and activate it
    '''bash 
    - python3 -m venv mnist_hdr-env 
    - source mnist_hdr-env/bin/activate

- Install the following python modules
    '''bash
    - pip install tensorflow matplotlib numpy
    
- Open the Handwritten_digit_recognition_assignment file 
    '''bash
    - cd .../Handwritten_digit_recognition_assignment
        *where '...' is the file path that leads to Handwritten_digit_recognition_assignment

## How to build the model
- Build the model via this command
    '''bash
    - python -i mnist_digit_recognition.py
        *This will: load the model mnist_model.h5 since it already exists in the file, otherwise it would create a new one
                    evaluate the model on test data
                    show training history and prediction samples
                    generate training_history.png and predictions,png but they are already in the file so it won't generate again
                    run the program in interactive mode allowing the user to interact with the functions and variables in the script without needing to import them

## How to now predict a digit from an image of any file type (jpeg, png, webp...)
- Load the model created in the mnist_digit_recognition.py script
    '''python
    - model = load_model('mnist_model.h5')
- Predict a digit from an image
    '''python
    - predict_own_digit(model, "image_path") [image_path is the file path of where the image the user wants to have predicted is located]
        *image must contain single digit
        *for best results, use image where digit is centered, written in black on a white background
        *program will automatically resize and preprocess your image
        *image files 1.webp, 2.webp and 3.webp contained in file for user's convenience for testing
    - View prediction results on the console as well as time taken to predict
