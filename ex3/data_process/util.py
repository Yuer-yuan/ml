import numpy as np
import struct
import os
import matplotlib.pyplot as plt

def load_minist(file_dir, is_images=True):
    bin_file_name=os.listdir(file_dir)[0]
    with open(os.path.join(file_dir, bin_file_name), 'rb') as bin_file:
        if is_images:
            fmt_header = '>iiii'
            magic, num_images, num_rows, num_cols = struct.unpack(fmt_header, bin_file.read(16))
            data = np.fromfile(bin_file, dtype=np.uint8).reshape(num_images, 1, num_rows, num_cols).transpose(0, 2, 3, 1)
        else:
            fmt_header = '>ii'
            magic, num_images = struct.unpack(fmt_header, bin_file.read(8))
            data = np.fromfile(bin_file, dtype=np.uint8).reshape(num_images, 1)
    print('Load images from %s, number: %d, data shape: %s' % (file_dir, num_images, str(data.shape)))
    return data


def load_data(mnist_dir, train_data_dir, train_label_dir, test_data_dir, test_label_dir):
    print('Loading MNIST data from files...')
    train_images = load_minist(os.path.join(mnist_dir, train_data_dir), True)
    train_labels = load_minist(os.path.join(mnist_dir, train_label_dir), False)
    test_images = load_minist(os.path.join(mnist_dir, test_data_dir), True)
    test_labels = load_minist(os.path.join(mnist_dir, test_label_dir), False)
    return train_images, train_labels, test_images, test_labels


def binarize(x, threshold=40):
    x[x<=threshold]=0
    x[x>threshold]=1
    return x;


def show_image(X, y):
    print(y)
    plt.imshow(X, cmap='gray' if X.shape[2] == 1 else None)


def one_hot(y, n_values):
    return np.eye(n_values)[y.flatten()]


def reverse_one_hot(y):
    return np.argmax(y, axis=-1)


def generate_regions(X, dim, stride):
    '''
    Generate regions of size dim x dim from X with stride.
    X is of shape (batch_size, height, width, in_nchannel)
    '''
    assert X.shape[1] >= dim
    assert X.shape[2] >= dim
    for fh, h in enumerate(range(0, X.shape[1] - dim + 1, stride)):
        for fw, w in enumerate(range(0, X.shape[2] - dim + 1, stride)):
            yield fh, fw, np.s_[:, h:h + dim, w:w + dim, :]


def softmax(x, axis = -1, epsilon = 1e-10):
    x = x - np.max(x, axis = axis, keepdims = True)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x, axis = axis, keepdims = True)
    return softmax_x