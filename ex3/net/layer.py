import numpy as np
from numpy import newaxis
from data_process.util import generate_regions, softmax

class Layer:
    def forward(self, x):
        self.x = x.copy()

    def backward(self, grad_in):
        raise NotImplementedError


class Conv(Layer):
    def __init__(self, out_nchannel, filter_size, stride = 1, padding = 0):
        self.out_nchannel = out_nchannel
        self.filter_size = filter_size
        self.stride = stride
        self.padding = padding
        self.W = None
        self.b = None

    def forward(self, x):
        '''
        x is of shape (batch_size, height, width, in_nchannel)
        '''
        super().forward(x)
        out_nchannel, filter_size, stride, padding = self.out_nchannel, self.filter_size, self.stride, self.padding
        if self.W is None:
            self.W = np.random.randn(
                filter_size, filter_size, x.shape[-1], out_nchannel) * np.sqrt(
                    2 / (x.shape[1] * x.shape[2] * x.shape[3]))
            self.b = np.zeros(out_nchannel)
        W, b = self.W, self.b

        x = np.pad(x, [(0, 0), (padding, padding), (padding, padding), (0, 0)], 'constant')
        Wx = np.zeros((len(x), (x.shape[1] - W.shape[0]) // stride + 1,
                       (x.shape[2] - W.shape[1]) // stride + 1, out_nchannel))

        for fh, fw, slice in generate_regions(x, filter_size, stride):
            Wx[:, fh, fw, :] = np.tensordot(x[slice], W, axes=3)
        return Wx + b

    def backward(self, grad_in):
        x = self.x
        dx = np.zeros_like(x, dtype=float)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)
        filter_size, stride, padding = self.filter_size, self.stride, self.padding
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding), (0, 0)), 'constant')
        dx_pad = np.zeros_like(x, dtype=float)

        for fh, fw, slice in generate_regions(x, filter_size, stride):
            grad_in_slice = grad_in[:, fh, fw, newaxis, newaxis, newaxis, :]
            dx_pad[slice] += np.sum(self.W * grad_in_slice, axis=-1)
            dW += np.sum(x[slice][..., newaxis] * grad_in_slice, axis=0)
            db += np.sum(grad_in_slice, axis=0).squeeze()
        dx = dx_pad[:, padding:-padding, padding:-padding, :] if padding > 0 else dx_pad
        self.W -= dW * self.learn_rate
        self.b -= db * self.learn_rate
        return dx

    
class ReLU(Layer):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        super().forward(x)
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, grad_in):
        dx = grad_in.copy()
        dx[self.mask] = 0
        return dx


class MaxPool(Layer):
    def __init__(self, filter_size, stride):
        self.filter_size = filter_size
        self.stride = stride

    def forward(self, x):
        '''
        x is of shape (batch_size, height, width, in_nchannel)
        '''
        super().forward(x)
        filter_size, stride = self.filter_size, self.stride
        out = np.zeros((len(x), (x.shape[1] - filter_size) // stride + 1,
                        (x.shape[2] - filter_size) // stride + 1, x.shape[3]))
        for fh, fw, slice in generate_regions(x, filter_size, stride):
            out[:, fh, fw, :] = np.max(x[slice], axis=(1, 2))
        return out

    def backward(self, grad_in):
        x = self.x
        filter_size, stride = self.filter_size, self.stride
        dx = np.zeros_like(x, dtype=float)
        for fh, fw, slice in generate_regions(x, filter_size, stride):
            xs = x[slice]
            indices = np.indices((xs.shape[0], xs.shape[-1]))
            max_indices = (indices[0], ) + np.unravel_index(
                xs.reshape((xs.shape[0], -1, xs.shape[-1])).argmax(axis=1),
                xs.shape[1:-1]) + (indices[1], )
            mask = np.zeros_like(xs)
            mask[max_indices] = 1
            dx[slice] += mask * grad_in[:, fh, newaxis, fw, newaxis, :]
        return dx


class Flatten(Layer):
    def forward(self, x):
        '''
        x is of shape (batch_size, height, width, in_nchannel)
        out is of shape (batch_size, -1)
        '''
        super().forward(x)
        return x.reshape((len(x), -1))

    def backward(self, grad_in):
        return grad_in.reshape(self.x.shape)


class Dense(Layer):
    def __init__(self, out_nchannel):
        self.out_nchannel = out_nchannel
        self.W = None
        self.b = None

    def forward(self, x):
        '''
        x is of shape (batch_size, in_nchannel)
        out is of shape (batch_size, out_nchannel)
        '''
        super().forward(x)
        in_nchannel = x.shape[-1]
        out_nchannel = self.out_nchannel
        if self.W is None:
            self.W = np.random.randn(in_nchannel, out_nchannel) * np.sqrt(
                2 / in_nchannel)
            self.b = np.zeros(out_nchannel)
        W, b = self.W, self.b
        return np.dot(x, W) + b

    def backward(self, grad_in):
        x = self.x
        dx = np.dot(grad_in, self.W.T)
        self.W -= np.dot(x.T, grad_in) * self.learn_rate
        self.b -= np.sum(grad_in, axis=0) * self.learn_rate
        return dx


class SoftmaxCrossEntropy(Layer):
    def __init__ (self):
        self.grad = None

    def forward(self, x, y):
        '''
        x is of shape (batch_size, in_nchannel)
        y is of shape (batch_size, in_nchannel)
        return probs of shape (batch_size, in_nchannel) and loss
        '''
        super().forward(x)
        m = x.shape[0]
        probs = softmax(x)
        loss = (-1 / m) * np.log(probs[y==1]).sum()
        self.grad = (probs - y) / m
        return probs, loss

    def backward(self, grad_in):
        return self.grad