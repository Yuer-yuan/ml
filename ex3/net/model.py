import numpy as np
from data_process.util import softmax, reverse_one_hot
from tqdm import tqdm
from sklearn import metrics

class Model():
    def __init__(self, layers, learn_rate = 0.001):
        for layer in layers:
            layer.learn_rate = learn_rate
        self.layers = layers
        self.learn_rate = learn_rate

    def forward(self, x, y = None):
        '''
        x is of shape (batch_size, height, width, in_nchannel)
        y is of shape (batch_size, 10)
        if y is None, return probs of shape (batch_size, 10)
        if y is not None, return probs of shape (batch_size, 10) and loss
        '''
        n_samples = x.shape[0]
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        if y is None:
            probs = softmax(x)
            return probs
        assert len(y) == n_samples
        probs, loss = self.layers[-1].forward(x, y)
        return probs, loss

    def backward(self):
        '''
        Backward propagation and update weight of each layer

        '''
        grad = 1
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


    def fit(self, x_train, y_train, x_validate, y_validate, epochs=1, batch_size=32):
        '''
        1. train the model by epochs and batches
        2. print validate result of each epoch

        '''
        n_sample = len(x_train)
        n_batch = (n_sample - 1) // batch_size + 1
        acc_list = []
        loss_list = []
        grad_list = []
        for epoch in range(epochs):
            print(f"Epoch {epoch+1} ================")
            with tqdm(total=n_batch) as t:
                total_loss = total_acc = 0
                for i in range(n_batch):
                    batch = range(batch_size * i, min(batch_size * (i + 1), n_sample))
                    probs, loss = self.forward(x_train[batch], y_train[batch])
                    acc = (1 / len(batch)) * np.sum(reverse_one_hot(probs) == reverse_one_hot(y_train[batch]))
                    grad = self.backward()
                    total_loss += loss
                    total_acc += acc
                    acc_list.append(acc)
                    loss_list.append(loss)
                    grad_list.append(grad)
                    if (i + 1) % 32 == 0 or i + 1 == n_batch:
                        t.set_postfix({
                            'avg_loss': total_loss / (i + 1),
                            'avg_accuracy': total_acc / (i + 1),
                            'max_abs_gradient': np.max(abs(grad))
                        })
                        cur_n_batch = i % 32 + 1
                        t.update(cur_n_batch)
            print("Validation:")
            validate_probs, validate_loss = self.evaluate(x_validate, y_validate)
            print('loss: ', validate_loss)
            print(metrics.classification_report(reverse_one_hot(validate_probs), reverse_one_hot(y_validate)))

    def predict(self, x, batch_size=32):
        '''
        return probabilities
        '''
        probs = []
        n_sample = len(x)
        n_batch = (n_sample - 1) // batch_size + 1
        for i in tqdm(range(n_batch)):
            batch = range(batch_size * i, min(batch_size * (i + 1), n_sample))
            probs.append(self.forward(x[batch]))
        return np.concatenate(probs)

    def evaluate(self, x, y, batch_size=32):
        '''
        return probabilities and average loss of each batch
        '''
        probs = []
        n_sample = len(x)
        n_batch = (n_sample - 1) // batch_size + 1
        total_loss = 0
        for i in tqdm(range(n_batch)):
            batch = range(batch_size * i, min(batch_size * (i + 1), n_sample))
            temp_probs, loss = self.forward(x[batch], y[batch])
            probs.append(temp_probs)
            total_loss += loss
        return np.concatenate(probs), total_loss / n_batch