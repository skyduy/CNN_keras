#!/usr/bin/env python
# encoding: utf-8

"""
    File name: predict_1.py
    Function Des: ...
    ~~~~~~~~~~

    author: Skyduy <cuteuy@gmail.com> <http://skyduy.me>

"""
import urllib
import numpy as np
from utils import asc_chr
from PIL import Image
from keras.models import Graph
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D

print '... loading data'


def get_data():
    pic_name = 'tmp.jpg'
    urllib.urlretrieve('http://passport2.chaoxing.com/img/code', 'tmp/%s' % pic_name)
    results = {}
    raw_img = Image.open('tmp/tmp.jpg')
    img = raw_img.convert('1')
    x_size, y_size = img.size
    for x in xrange(x_size):
        results.setdefault(x, 0)
        for y in xrange(y_size):
            pixel = img.getpixel((x, y))
            if pixel == 0:
                results[x] += 1
    begin = end = -1
    for i in range(150):
        if results[i] > 0:
            begin = i
            break
    threshold_list = []
    for i in range(131, 134):
        threshold_list.append(results[i])
    threshold = max(threshold_list)
    i = 150
    stop = False
    while not stop:
        i -= 1
        if i == -1:
            begin, end = 10, 135
            break
        v = results[i]
        if v >= max((3.5, 2 * threshold)):
            stop = True
            tmp_value = v
            while True:
                i += 1
                if results[i] <= tmp_value:
                    if results[i] <= threshold:
                        end = i
                        break
                    else:
                        tmp_value = results[i]
                else:
                    end = i - 1
    centers = [((2 * i + 1) * end + (9 - 2 * i) * begin) / 10 for i in range(5)]
    data = np.empty((5, 3, 40, 40), dtype="float32")
    for index, center in enumerate(centers):
        a = center - 19
        b = center + 21
        img = raw_img.crop((a, 0, b, y_size))
        arr = np.asarray(img, dtype="float32") / 255.0
        data[index, :, :, :] = np.rollaxis(arr, 2)
    return data

X = get_data()


print '... create model'

graph = Graph()
graph.add_input(name='input', input_shape=(3, 40, 40))
graph.add_node(Convolution2D(32, 9, 9, activation='relu'), name='conv1', input='input')
graph.add_node(Convolution2D(32, 9, 9, activation='relu'), name='conv2', input='conv1')
graph.add_node(MaxPooling2D(pool_size=(2, 2)), name='pool', input='conv2')
graph.add_node(Dropout(0.25), name='drop', input='pool')
graph.add_node(Flatten(), name='flatten', input='drop')
graph.add_node(Dense(640, activation='relu'), name='ip', input='flatten')
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

graph.load_weights('model/model_1.hdf5')
print '... predicting'
result = graph.predict({'input': X})
out = [np.argmax(result['out'][0]), np.argmax(result['out'][1]),
       np.argmax(result['out'][2]), np.argmax(result['out'][3]),
       np.argmax(result['out'][4])]

print out

r = [asc_chr(i) for i in out]

print r
