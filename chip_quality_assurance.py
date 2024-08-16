#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 13:14:55 2024

@author: meghanapuli
"""

'''
Problem statement

Build a regularized logistic regression model to predict whether microchips from a fabrication plant 
passes quality assurance (QA). During QA, each microchip goes through various tests to ensure it is 
functioning correctly. 

Suppose you are the product manager of the factory and you have the test results for some microchips on 
two different tests. 
- From these two tests, you would like to determine whether the microchips should be accepted or rejected. 
- To help you make the decision, you have a dataset of test results on past microchips, from which you can build a logistic regression model.
y_train = 1` if the microchip was accepted 
y_train = 0` if the microchip was rejected 
'''

import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data = np.loadtxt("chip_test_results.txt", delimiter=',')
    X = data[:,:2]
    y = data[:,2]
    return X, y

# Load the dataset
X_train, y_train = load_data()

def plot_data(X, y, pos_label="y=1", neg_label="y=0"):
    positive = y == 1
    negative = y == 0
    
    # Plot examples
    plt.plot(X[positive, 0], X[positive, 1], 'k+', label=pos_label)
    plt.plot(X[negative, 0], X[negative, 1], 'yo', label=neg_label)
    
# Plot training data
print("\nTraining data")
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")

# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()

print("\nThe plot shows that our dataset cannot be separated into positive and negative examples by a straight-line through the plot. Therefore, a straight forward application of logistic regression will not perform well on this dataset since logistic regression will only be able to find a linear decision boundary.\n")

def map_feature(X1, X2):
    """
    Feature mapping function to polynomial features    
    """
    X1 = np.atleast_1d(X1)
    X2 = np.atleast_1d(X2)
    degree = 6
    out = []
    for i in range(1, degree+1):
        for j in range(i + 1):
            out.append((X1**(i-j) * (X2**j)))
    return np.stack(out, axis=1)

print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)

#print("\nX_train[0]:", X_train[0])
#print("mapped X_train[0]:", mapped_X[0])

def sigmoid(z):
      
    g = 1.0/(1.0+np.exp(-z))

    return g

# Predict function to produce 0 or 1 predictions given a dataset
def predict(X, w, b): 

    m, n = X.shape   
    p = np.zeros(m)

    # Loop over each example
    for i in range(m):   
        z_wb = 0
        # Loop over each feature
        for j in range(n): 
            # Add the corresponding term to z_wb
            z_wb += X[i,j] * w[j]
        
        # Add bias term 
        z_wb += b
        
        # Calculate the prediction for this example
        f_wb = sigmoid(z_wb)

        # Apply the threshold
        if f_wb >= 0.5:
            p[i] = 1
        else:
            p[i] = 0

    return p

# Compute the prediction of the model
def compute_model_output(X, w, b): 
    z = np.dot(w,X) + b
    f_wb = sigmoid(z)
    return f_wb

# Compute the cost of the model
def compute_cost(X, y, w, b, lambda_ = 1):

    m, n = X.shape

    cost = 0
    reg_cost = 0
    for i in range(m):
        z_i = np.dot(w,X[i]) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i]*np.log(f_wb_i) - (1 - y[i])*np.log(1 - f_wb_i)
    cost = cost / m
    
    for j in range(n):
        reg_cost += w[j]**2
    
    reg_cost *= (lambda_)/(2*m)
    
    total_cost = cost + reg_cost
    return total_cost

# Compute the gradient
def compute_gradient(X, y, w, b, lambda_ = 1): 

    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = np.dot(w,X[i]) + b
        f_wb = sigmoid(z_wb)
        err_i = f_wb - y[i]
        
        for j in range(n):
            dj_dw[j] = dj_dw[j] + err_i * X[i,j]
        
        dj_db += err_i
    
    dj_dw = dj_dw / m
    dj_db = dj_db / m
    
    for j in range(n):
        dj_dw[j] +=  (lambda_ / m) * w[j]

    return dj_db, dj_dw

# Gradient descent to find optimal w,b
def gradient_descent(X, y, w_in, b_in, gradient_function, alpha, num_iters, lambda_): 

    print("\nCost vs iterations")
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db 
        
        if i % 1000 == 0 or i == num_iters-1:
            print(f"Iteration {i:4d}:", "Cost:", round(compute_cost(X, y, w_in, b_in, lambda_),2))
            
    return w_in, b_in

# Initialize fitting parameters
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
w_tmp  = np.random.rand(X_mapped.shape[1])-0.5
b_tmp  = 1.

lambda_ = 0.01 

alpha = 0.01
num_iters = 10000

w_out, b_out = gradient_descent(X_mapped, y_train, w_tmp, b_tmp, compute_gradient, alpha, num_iters, lambda_) 

print(f"\nOptimal parameters: w:{ w_out}, b:{ b_out}")

cost = compute_cost(X_mapped, y_train, w_out, b_out, lambda_)
print("\nCost of our model:", round(cost,2))

def plot_decision_boundary(w, b, X, y):

    plot_data(X[:, 0:2], y)
    
    if X.shape[1] <= 2:
        plot_x = np.array([min(X[:, 0]), max(X[:, 0])])
        plot_y = (-1. / w[1]) * (w[0] * plot_x + b)
        
        plt.plot(plot_x, plot_y, c="b")
        
    else:
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)
        
        z = np.zeros((len(u), len(v)))

        for i in range(len(u)):
            for j in range(len(v)):
                z[i,j] = sigmoid(np.dot(map_feature(u[i], v[j]), w) + b)
     
        z = z.T

        plt.contour(u,v,z, levels = [0.5], colors="g")

print("\nDecision boundary")
plot_decision_boundary(w_out, b_out, X_mapped, y_train)
# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()

# Compute accuracy on our training set
p = predict(X_mapped, w_out, b_out)
print('\nTrain Accuracy: %f'%(np.mean(p == y_train) * 100))

print("\nPlease enter the Microchip test scores")
X_test = []
X_test.append(float(input("\nTest 1 score: ")))
X_test.append(float(input("\nTest 2 score: ")))

X_test_mapped = map_feature(X_test[0], X_test[1])
#print(X_test_mapped)

computed_value = compute_model_output(X_test_mapped[0], w_out, b_out)
print("\nProbability of chip's acceptance: ", round(computed_value,2))

if computed_value < 0.5:
    output = 0
    print("Result: Rejected")
    
else:
    output = 1
    print("Result: Accepted")
