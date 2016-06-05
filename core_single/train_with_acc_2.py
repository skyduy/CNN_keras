#!/usr/bin/env python
# encoding: utf-8

"""
    File name: train_with_acc_2.py
    Function Des: 效果跟1差不多，但速度比1快。
    ~~~~~~~~~~

    author: Skyduy <cuteuy@gmail.com> <http://skyduy.me>

"""
from numpy import argmax, array
from sklearn.cross_validation import train_test_split
from keras.callbacks import Callback, ModelCheckpoint
from keras.models import Graph
from keras.utils import np_utils
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D

from utils import load_single_data

print '... loading data'
X, Y_all = load_single_data()

X_train, X_test, y_train, y_test = train_test_split(X, Y_all, test_size=0.1, random_state=0)


y_train = np_utils.to_categorical(y_train, 19)

graph = Graph()
graph.add_input(name='input', input_shape=(3, 40, 40))
graph.add_node(Convolution2D(22, 5, 5, activation='relu'), name='conv1', input='input')
graph.add_node(MaxPooling2D(pool_size=(2, 2)), name='pool1', input='conv1')
graph.add_node(Convolution2D(44, 3, 3, activation='relu'), name='conv2', input='pool1')
graph.add_node(MaxPooling2D(pool_size=(2, 2)), name='pool2', input='conv2')
graph.add_node(Dropout(0.25), name='drop', input='pool2')
graph.add_node(Flatten(), name='flatten', input='drop')
graph.add_node(Dense(256, activation='relu'), name='ip', input='flatten')
graph.add_node(Dropout(0.5), name='drop_out', input='ip')
graph.add_node(Dense(19, activation='softmax'), name='result', input='drop_out')

graph.add_output(name='out', input='result')

print '... compiling'
graph.compile(
    optimizer='adadelta',
    loss={
        'out': 'categorical_crossentropy',
    }
)
print '... training'


class ValidateAcc(Callback):
    def on_epoch_end(self, epoch, logs={}):
        print '\n————————————————————————————————————'
        graph.load_weights('tmp/weights.%02d.hdf5' % epoch)
        r = graph.predict({'input': X_test}, verbose=0)
        y_predict = array([argmax(i) for i in r['out']])
        length = len(y_predict) * 1.0
        acc = sum(y_predict == y_test) / length
        print 'Single picture test accuracy: %2.2f%%' % (acc * 100)
        print 'Theoretical accuracy: %2.2f%% ~  %2.2f%%' % ((5*acc-4)*100, pow(acc, 5)*100)
        print '————————————————————————————————————'

check_point = ModelCheckpoint(filepath="tmp/weights.{epoch:02d}.hdf5")
back = ValidateAcc()
print 'Begin train on %d samples... test on %d samples...' % (len(y_train), len(y_test))
graph.fit(
    {'input': X_train, 'out': y_train},
    batch_size=128, nb_epoch=100, callbacks=[check_point, back]
)
print '... saving'
graph.save_weights('model/model_2.hdf5')
