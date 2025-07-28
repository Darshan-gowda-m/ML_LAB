import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def lwr(x,y,x0,tau):
    m=len(x)
    X=np.c_[np.ones(m),x]
    X0=np.array([1,x0])
    w=np.exp(-((x-x0)**2)/(2*tau**2))
    W=np.diag(w)
    thetha=np.linalg.pinv(X.T @ W @ X)@ X.T @ W @ y
    return X0@ thetha


np.random.seed(0)
x=np.linspace(0,10,100)
y=np.sin(x)+0.3*np.random.randn(100)

x_test=np.linspace(0,10,500)
y_test=np.array([lwr(x,y,x0,tau=0.3) for x0 in x_test])


plt.figure(figsize=(8,4))

plt.plot(x_test,y_test,color='green')
plt.scatter(x,y,color='yellow')
plt.title('LWR')
plt.show()
