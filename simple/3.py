import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("train.csv")
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode categorical features
le = LabelEncoder()
X = X.apply(le.fit_transform)
y = le.fit_transform(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Train decision tree
model = DecisionTreeClassifier(criterion="entropy")
model.fit(X_train, y_train)

# Predict & print results
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Predictions:", y_pred.tolist())
print(export_text(model, feature_names=list(X.columns)))
