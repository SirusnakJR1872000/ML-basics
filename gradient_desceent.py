# Implementing Gradient Descent for Optimizing a Cost Function

import numpy as np

def gradient_decent(x, y, learning_rate = 0.01, num_iterations = 100):
    # take the number of training samples
    m = len(y)
    # Initializes the parameters (intercept and slope) to zeros
    theta = np.zeroes(2)
    # now we will iterate through the gradient descent
    # we will go till the iterations
    for i in range(num_iterations):
        predictions = theta[0] + theta[1] * x
        errors = predictions - y
        theta[0] -= learning_rate * (1 / m) * np.sum(errors)
        theta[1] -= learning_rate * (1 / m) * np.sum(errors) * x
    return theta

# Example usage
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 1.5, 3.5, 2.5])

theta = gradient_decent(x, y, learning_rate = 0.01, num_iterations = 100)
