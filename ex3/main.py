from data_process.util import load_data, binarize, one_hot
from net.model import Model
from net.layer import Conv, ReLU, MaxPool, Flatten, Dense, SoftmaxCrossEntropy
import numpy as np

if __name__ == '__main__':
    # load data
    mnist_dir = "mnist_data/"
    train_data_dir = "train-images-idx3-ubyte"
    train_label_dir = "train-labels-idx1-ubyte"
    test_data_dir = "t10k-images-idx3-ubyte"
    test_label_dir = "t10k-labels-idx1-ubyte"
    train_images, train_labels, test_images, test_labels = load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir)
    
    # binarize images and encode labels to one-hot
    train_images = binarize(train_images)
    test_images = binarize(test_images)
    train_labels = one_hot(train_labels, n_values = 10)
    test_labels = one_hot(test_labels, n_values = 10)

    # create model
    np.random.seed(0)
    model = Model([
        Conv(out_nchannel = 6, filter_size = 5, stride = 1, padding = 2),
        ReLU(),
        MaxPool(filter_size = 2, stride = 2),
        Conv(out_nchannel = 16, filter_size = 5, stride = 1, padding = 0),
        ReLU(),
        MaxPool(filter_size = 2, stride = 2),
        Flatten(),
        Dense(out_nchannel = 120),
        ReLU(),
        Dense(out_nchannel = 84),
        ReLU(),
        Dense(out_nchannel = 10),
        SoftmaxCrossEntropy()
    ], learn_rate=0.02)

    # train and validate, print the validate data
    model.fit(x_train=train_images, y_train=train_labels, x_validate=test_images, y_validate=test_labels, epochs=2, batch_size=32)



    
    