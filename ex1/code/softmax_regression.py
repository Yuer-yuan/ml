# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt

def softmax(z):
    z-=np.max(z)
    softmax=(np.exp(z) / np.sum(np.exp(z), axis=0))
    return softmax

def get_gradient(x, y, theta, lam):
    m = x.shape[0]
    score = np.dot(theta, x.T)
    softmax_score = softmax(score)
    f=-np.sum(y*np.log(softmax_score))/m+lam*np.sum(theta*theta)/2
    g = np.dot(softmax_score-y,x)/m+lam*theta
    return f, g

def softmax_regression(theta, x, y, iters, alpha, lam):
    # TODO: Do the softmax regression by computing the gradient and 
    # the objective function value of every iteration and update the theta
    losses=[]
    for iter in range(iters):
        f, g=get_gradient(x,y,theta,lam)
        theta-=alpha*g
        losses.append(f)
    plt.plot(losses)
    plt.show()
    return theta
    
