import numpy as np

# Sigmoid activation and derivative
def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_deriv(x): return x*(1-x)

# Training data (X: features, y: labels)
X = np.array([[0,0],[0,1],[1,0],[1,1]])  # XOR input
y = np.array([[0],[1],[1],[0]])          # XOR output

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.rand(2,2)   # input -> hidden
b1 = np.random.rand(1,2)
W2 = np.random.rand(2,1)   # hidden -> output
b2 = np.random.rand(1,1)

# Training with Backpropagation
lr, epochs = 0.5, 10000
for _ in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Backpropagation
    error = y - a2
    d2 = error * sigmoid_deriv(a2)
    d1 = np.dot(d2, W2.T) * sigmoid_deriv(a1)

    # Weight updates
    W2 += np.dot(a1.T, d2) * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr
    W1 += np.dot(X.T, d1) * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr

# Testing
print("Predictions after training:")
print(np.round(a2))
