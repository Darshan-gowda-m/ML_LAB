import numpy as np

# Sigmoid activation and derivative
def sigmoid(x): return 1/(1+np.exp(-x))
def sigmoid_deriv(x): return x*(1-x)

# Training data (XOR problem)
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.rand(2,2)   # input -> hidden
b1 = np.random.rand(1,2)
W2 = np.random.rand(2,1)   # hidden -> output
b2 = np.random.rand(1,1)

# Hyperparameters
epochs, lr = 10000, 0.5   # ✅ fixed order

for epoch in range(1, epochs+1):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    # Compute error
    error = y - a2
    loss = np.mean(np.square(error))

    # Backpropagation
    d2 = error * sigmoid_deriv(a2)
    d1 = np.dot(d2, W2.T) * sigmoid_deriv(a1)

    # Update weights
    W2 += np.dot(a1.T, d2) * lr
    b2 += np.sum(d2, axis=0, keepdims=True) * lr
    W1 += np.dot(X.T, d1) * lr
    b1 += np.sum(d1, axis=0, keepdims=True) * lr

    # Print progress every 1000 epochs
    if epoch % 1000 == 0 or epoch == 1:
        print(f"\nEpoch {epoch}")
        print("Hidden layer output:\n", a1)
        print("Final output:\n", a2)
        print("Error:\n", error)
        print("Loss:", loss)

# Final predictions
print("\nFinal Predictions:")
print(np.round(a2))
