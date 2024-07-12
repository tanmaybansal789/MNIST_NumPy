import matplotlib.pyplot as plt
import numpy as np
import os
from mnist import MNIST
from PIL import Image, ImageDraw
import tkinter as tk



def sigmoid(x): return 1/(1 + np.exp(-x)) #Sigmoid activation function

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
    #Softmax activation function for output 

def cross_entropy_loss(y_pred, y_true): return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0] # Assuming y_true is one-hot encoded and y_pred is the output probabilities from the softmax


def load_dataset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    relative_mndata_path = os.path.join(script_dir, "MNIST_files")
    mndata = MNIST(relative_mndata_path)
    x_train, y_train = mndata.load_training()
    x_test, y_test = mndata.load_testing()
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def load_model():
    weight_matrices = []
    biases = []
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define the relative path
    relative_weight_path = os.path.join(script_dir, "100_epochs/weight_matrices.npz")
    relative_bias_path = os.path.join(script_dir, "100_epochs/biases.npz")
    # Load weight matrices
    with np.load(relative_weight_path) as data:
        for key in data.keys():
            weight_matrices.append(data[key])

    # Load biases
    with np.load(relative_bias_path) as data:
        for key in data.keys():
            biases.append(data[key])

    return weight_matrices, biases


def forward_propogation(weight_matrices, input_layer, biases):
    activations = [input_layer] # Add input layer as first item in activations
    for layer in range(len(weight_matrices)):
        z = np.matmul(weight_matrices[layer], activations[layer])
        if layer != len(weight_matrices) - 1: 
            activated = sigmoid(z + biases[layer]) # Sigmoid activation function
        else:
            activated = softmax(z + biases[layer]) # Softmax activation function for output layer
        activations.append(activated)
    
    return activations    


weight_matrices, biases = load_model()

print("Model shape:", *[matrix.shape[1] for matrix in weight_matrices], weight_matrices[-1].shape[0])

def predict_digit(in_digit):
    activations = forward_propogation(weight_matrices, in_digit, biases)
    predicted_digit = np.argmax(activations[-1])
    return predicted_digit, activations[-1][predicted_digit] * 100

_, _, x_test, y_test = load_dataset()
x_test = x_test.astype('float32') / 255.0

print("Test data shape: ", x_test.shape)

for digit, label in zip(x_test, y_test):
    plt.imshow(digit.reshape((28,28)))
    plt.show()
    predicted_digit, percent_sure = predict_digit(digit)
    print(f"Actual: {label}\nPredicted Digit: {predicted_digit}\nPercent Sure:{percent_sure:0.2f}%")
    input()
