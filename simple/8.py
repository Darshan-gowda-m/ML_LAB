from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris=load_iris()
x,y=iris.data,iris.target;

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42,shuffle=True)

k=KNeighborsClassifier(n_neighbors=3)
k.fit(x_train,y_train)

print(f"features={iris.feature_names}\n target={iris.target}\n");


y_pred=k.predict(x_test)
a2=accuracy_score(y_pred,y_test)

print(f"testing accuracy ={a2}\n")
print(f"predicted={y_pred}\nactual={y_test}\n")
print(f"total number of misclassified examples are :{sum(p!=t for p,t in zip(y_pred,y_test))}")

