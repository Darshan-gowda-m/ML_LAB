import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
from colorama import init
from sklearn.model_selection import train_test_split

init(autoreset=True)

age = {'Super': 0, 'old': 1, 'adult': 2, 'teen': 3}
gender = {'male': 0, 'female': 1}
cholestrol = {'high': 0, 'moderate': 1, 'low': 2}
diet = {'low': 0, 'moderate': 1, 'high': 2}

x = np.array([
    [0, 1, 0, 1],  # Yes
    [1, 0, 0, 0],  # Yes
    [2, 0, 0, 0],  # Yes
    [1, 1, 0, 1],  # Yes
    [2, 1, 0, 0],  # Yes
    [3, 0, 1, 2],  # No
    [4, 1, 1, 2],  # No
    [4, 1, 1, 2],  # No
    [3, 1, 1, 2],  # No
    [2, 0, 1, 2]   # No
])
y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1
])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
model = MultinomialNB().fit(x_train, y_train)
y_pred = model.predict(x_test)

print("\n" + "="*25 + " Evaluation " + "="*25)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}\n")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

while True:
    try:
        u = [
            int(input(f"Age {age}: ")),
            int(input(f"Gender {gender}: ")),
            int(input(f"Cholestrol {cholestrol}: ")),
            int(input(f"Diet {diet}: "))
        ]

        p = model.predict_proba([u])[0]
        print(f"no: {p[1]:.2f}\nyes: {p[0]:.2f}")
        print("prediction:", "yes" if p[0] > p[1] else "no")

    except Exception as e:
        print("invalid input\n", e)

    if input("Enter 'y' to continue: ").lower() != 'y':
        break
