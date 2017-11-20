"""
    File name: train.py
    Function Des: ...

    ~~~~~~~~~~

    author: Skyduy <cuteuy@gmail.com> <http://skyduy.me>

"""

from utils import load_data
from sklearn.model_selection import train_test_split

print('... loading data')
X, Y = load_data('../samples')

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=0)
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)

# TODO to be continue
