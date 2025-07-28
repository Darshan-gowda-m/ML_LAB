

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# === Load Iris dataset ===
iris = load_iris()
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, shuffle=True)

# Train k-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Results
print(f"Feature Names: {iris.feature_names}")
print(f"Target Names: {iris.target_names}")
print(f"\nAccuracy: {accuracy:.2f}")
print(f"\nPredicted: {y_pred}")
print(f"Actual:    {y_test}")

# Misclassifications
result = ["✅ Correct" if p == t else "❌ Wrong" for p, t in zip(y_pred, y_test)]
print("\nResult:", result)
print(f"Total misclassified samples: {sum(p != t for p, t in zip(y_pred, y_test))}")
