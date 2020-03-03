import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('https://raw.githubusercontent.com/mubaris/potential-enigma/master/student.csv')
print(data.shape)
data.head()

math = data['Math'].values
read = data['Reading'].values
write = data['Writing'].values

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(math,read,write,color='#ef1234')
plt.show()

m = len(math)
x0 = np.ones(m)
X = np.array([x0,math,read]).T  
B = np.array([0,0,0])
Y= np.array(write)
alpha = 0.0001


def cost_function(X,Y,B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) **2)/(2*m)
    return  J

initail_cost = cost_function(X,Y,B)
print(initail_cost)

def gradient_descent(X, Y, B,alpha, iterations):

    cost_history = [0] *iterations
    m = len(Y)

    for iterations in range(iterations):
        h = X.dot(B)
        loss = h - Y
        gradient = X.T.dot(loss)/m
        B = B - alpha * gradient

        cost = cost_function(X,Y,B)
        cost_history[iterations] = cost

    return B, cost_history

newB , cost_history = gradient_descent(X,Y,B,alpha, 100000)
print(newB)

print(cost_history[-1])

def predict(row, B):
    th = B[0]

    for i in range(len(row)):
        th += B[i+1] * row[i]
        print(B[i+1])
    return th

d = [48,68]
print(predict(d,newB))