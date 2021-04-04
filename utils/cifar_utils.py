# ECBM E4040 Neural Networks and Deep Learning
# This is a utility function to help you download the dataset and preprocess the data we use for this homework.
# requires several modules: _pickle, tarfile, glob. If you don't have them, search the web on how to install them.
# You are free to change the code as you like.

# Import modules. If you don't have them, try `pip install xx` or `conda
# install xx` in your console.
import pickle
import os
import tarfile
import urllib.request as url
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res


def download_data():
    """
    Download the CIFAR-100 data from the website, which is approximately 170MB.
    The data (a .tar.gz file) will be store in the ./data/ folder.
    :return: None
    """
    if not os.path.exists('./data'):
        os.mkdir('./data')
        print('Start downloading data...')
        url.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
                        "./data/cifar-100-python.tar.gz")
        print('Download complete.')
    else:
        if os.path.exists('./data/cifar-100-python.tar.gz'):
            print('CIFAR-100 package already exists.')


def load_data():
    """
    Unpack the CIFAR-100 dataset and load the coarse datasets (20-class).
    :return: A tuple of label_map, data/labels. For both training and test sets.
    """
    # If the data hasn't been downloaded yet, download it first.
    if not os.path.exists('./data/cifar-100-python.tar.gz'):
        download_data()
    # Check if the package has been unpacked, otherwise unpack the package
    if not os.path.exists('./data/cifar-100-python/'):
        package = tarfile.open('./data/cifar-100-python.tar.gz')
        package.extractall('./data')
        package.close()
        
    # Go to the location where the files are unpacked
    os.chdir('./')
    
    # load the label_map and data
    meta = unpickle('./data/cifar-100-python/meta')

    coarse_label_names = [t.decode('utf8') for t in meta[b'coarse_label_names']]

    train = unpickle('./data/cifar-100-python//train')
    y_train = np.array(train[b'coarse_labels'])
    X_train = np.array(train[b'data'])

    test = unpickle('./data/cifar-100-python//test')
    y_test = np.array(test[b'coarse_labels'])
    X_test = np.array(test[b'data'])
    
    return coarse_label_names, X_train, y_train, X_test, y_test
