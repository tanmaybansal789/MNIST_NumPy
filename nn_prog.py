#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from mnist import MNIST

def sigmoid(x): return 1/(1 + np.exp(-x)) #Sigmoid activation function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
    #Softmax activation function for output layer
def sigmoid_rate(a): return a * (1 - a) # a represents activated value.
def cross_entropy_loss(y_pred, y_true): return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0] # Assuming y_true is one-hot encoded and y_pred is the output probabilities from the softmax
def relu(x): return np.maximum(0, x)
def relu_rate(x): return np.where(x>0, 1, 0)


def load_dataset():
    mndata = MNIST("C:/Users/20bansalta/OneDrive - Hampton School/Documents/NN implementation/MNIST_files")
    x_train, y_train = mndata.load_training()
    x_test, y_test = mndata.load_testing()
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def forward_propogation(weight_matrices, input_layer, biases):
    """Performs dynamic forward propogation on a list of numpy arrays representing weight matrices and biases, with an input layer"""
    activations = [input_layer] # Add input layer as first item in activations
    for layer in range(len(weight_matrices)):
        z = np.matmul(weight_matrices[layer], activations[layer])
        if layer != len(weight_matrices) - 1: 
            activated = relu(z + biases[layer]) # Sigmoid activation function
        else:
            activated = softmax(z + biases[layer]) # Softmax activation function for output layer
        activations.append(activated)
    
    return activations    

def back_propogation(weight_matrices, biases, activations, y_true):
    """Performs backpropogation with cross entropy loss, ReLU and softmax on output layer"""
    dC_dW = [np.zeros(w.shape) for w in weight_matrices]  # Gradient of cost w.r.t. weights
    dC_db = [np.zeros(b.shape) for b in biases]          # Gradient of cost w.r.t. biases
    dC_dz = activations[-1] - y_true                     # Gradient of the loss w.r.t. the last layer's activations

    # Loop over the layers in reverse order
    for L in reversed(range(len(weight_matrices))):
        # Calculate the gradient of the cost w.r.t. biases (same as dC_dz for the last layer)
        dC_db[L] = dC_dz

        # The gradient w.r.t. weights is the outer product of dC_dz and activations[L]
        dC_dW[L] = np.outer(dC_dz, activations[L])

        if L > 0:  # If this isn't the input layer
            da_dz = relu_rate(activations[L])  # Derivative of the activation function
            dC_dz = np.dot(weight_matrices[L].T, dC_dz) * da_dz # Backpropagate the error

    return dC_dW, dC_db

def one_hot_encode(labels, num_classes=10):
    # Create an array of zeros with shape (len(labels), num_classes)
    one_hot = np.zeros((len(labels), num_classes))
    # Set the appropriate elements to one
    one_hot[np.arange(len(labels)), labels] = 1
    return one_hot

def update_weights(weight_matrices, dC_dW, learning_rate):
    for L in range(len(weight_matrices)): weight_matrices[L] -= learning_rate * dC_dW[L]
    #update weights based on a learning rate

def update_biases(biases, dC_db, learning_rate):
    for L in range(len(biases)): biases[L] -= learning_rate * dC_db[L]
    #update biases based on a learning rate

def train_network(weight_matrices, biases, x_train, y_train_one_hot, epochs, batch_size, learning_rate):
    """mini-batch training"""
    for epoch in range(epochs):
        # Shuffle the dataset at the beginning of each epoch
        permutation = np.random.permutation(x_train.shape[0])
        x_train_shuffled = x_train[permutation]
        y_train_shuffled = y_train_one_hot[permutation]

        # Iterate over mini-batches
        for i in range(0, x_train.shape[0], batch_size):
            x_batch = x_train_shuffled[i:i+batch_size]  # Extract the current mini-batch
            y_batch = y_train_shuffled[i:i+batch_size]

            # Initialize gradients for weights and biases
            batch_dC_dW = [np.zeros(w.shape) for w in weight_matrices]
            batch_dC_db = [np.zeros(b.shape) for b in biases]

            # Process each example in the mini-batch
            for x, y in zip(x_batch, y_batch):
                activations = forward_propogation(weight_matrices, x, biases)
                dC_dW, dC_db = back_propogation(weight_matrices, biases, activations, y)

                # Sum gradients for each example in the mini-batch
                for L in range(len(weight_matrices)):
                    batch_dC_dW[L] += dC_dW[L]
                    batch_dC_db[L] += dC_db[L]

            # Average the gradients over the mini-batch
            batch_dC_dW = [dW / batch_size for dW in batch_dC_dW]
            batch_dC_db = [db / batch_size for db in batch_dC_db]

            # Update weights and biases based on the average gradients from the mini-batch
            update_weights(weight_matrices, batch_dC_dW, learning_rate)
            update_biases(biases, batch_dC_db, learning_rate)

        # Print the epoch number to track progress
        print(f"Epoch {epoch + 1}/{epochs} completed")
                
def network_test(weight_matrices, biases, x_test, y_test):
    cumulative_loss = 0
    correct_predictions = 0

    for i in range(len(x_test)):
        input_layer = x_test[i]  # Single test example
        true_label = y_test[i]  # Corresponding true label

        # Perform forward propagation to get the prediction
        activations = forward_propogation(weight_matrices, input_layer, biases)
        prediction = np.argmax(activations[-1])  # The predicted class
        predicted_probs = activations[-1]  # The output probabilities from the softmax layer

        # Increment the correct predictions counter if the prediction matches the true label
        if prediction == np.argmax(true_label):  # As true_label is one-hot encoded
            correct_predictions += 1

        # Add the loss for the current test example to the cumulative loss
        cumulative_loss += cross_entropy_loss(predicted_probs, true_label)

    # Calculate the mean loss and accuracy
    mean_loss = cumulative_loss / len(x_test)
    accuracy = correct_predictions / len(x_test)

    return accuracy, mean_loss

np.random.seed(10)

weight_matrices = [np.random.randn(40, 784), np.random.randn(30, 40), np.random.randn(10, 30)]
biases = [np.random.randn(40), np.random.randn(30), np.random.randn(10)]

x_train, y_train, x_test, y_test = load_dataset()

# Normalize the data to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train_one_hot = one_hot_encode(y_train)
y_test_one_hot = one_hot_encode(y_test)

print("Training data shape: ", x_train.shape)
print("Training label shape: ", y_train_one_hot.shape)
print("Test data shape: ", x_test.shape)
print("Test label shape: ", y_test_one_hot.shape)

epochs = 10
batch_size = 32
learning_rate = 0.01


train_network(weight_matrices, biases, x_train, y_train_one_hot, epochs, batch_size, learning_rate)

accuracy, mean_loss = network_test(weight_matrices, biases, x_test, y_test_one_hot)

print(f"Accuracy:{accuracy}, Mean Loss:{mean_loss}")

np.savez("C:/Users/20bansalta/OneDrive - Hampton School/Documents/NN implementation/weight_matrices", *weight_matrices)
np.savez("C:/Users/20bansalta/OneDrive - Hampton School/Documents/NN implementation/biases.npz", *biases)
