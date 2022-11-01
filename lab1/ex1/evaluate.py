# coding=utf-8
import numpy as np


def predict(test_images, theta):
    # scores = np.dot(test_images, theta.T)
    # preds = np.argmax(scores, axis=1)
    scores = np.dot(theta, test_images.T)
    preds = np.argmax(scores, axis=0)
    return preds

def cal_accuracy(y_pred, y):
    # TODO: Compute the accuracy among the test set and store it in acc
    acc=0
    for i in range(y.shape[0]):
        if y_pred[i] == y[i]:
            acc += 1
    return acc/y.shape[0]