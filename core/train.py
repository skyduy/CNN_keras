"""
    File name: train.py
    Function Des:

    ~~~~~~~~~~

    author: Skyduy <cuteuy@gmail.com> <http://skyduy.me>

"""
from numpy import argmax, array
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import Callback, ModelCheckpoint

from utils import load_data, APPEARED_LETTERS


def prepare_data(folder):
    print('... loading data')
    letter_num = len(APPEARED_LETTERS)
    data, label = load_data(folder)
    data_train, data_test, label_train, label_test = \
        train_test_split(data, label, test_size=0.1, random_state=0)
    label_categories_train = to_categorical(label_train, letter_num)
    label_categories_test = to_categorical(label_train, letter_num)
    return ([data_train], [data_test],
            [label_categories_train[:, i] for i in range(5)],
            [label_categories_test[:, i] for i in range(5)])


def build_model():
    print('... construct network')
    inputs = layers.Input((40, 150, 3))
    common_layer = layers.Conv2D(32, 9, activation='relu')(inputs)
    common_layer = layers.Conv2D(32, 9, activation='relu')(common_layer)
    common_layer = layers.MaxPool2D((2, 2))(common_layer)
    common_layer = layers.Flatten()(common_layer)

    def _get_specific_layer():
        x = layers.Dropout(0.2)(common_layer)
        x = layers.Dense(640)(x)
        x = layers.Dropout(0.5)(x)
        out = layers.Dense(len(APPEARED_LETTERS), activation='softmax')(x)
        return out

    return Model(inputs=[inputs],
                 outputs=[_get_specific_layer() for _ in range(5)])


class TestAcc(Callback):
    def on_epoch_end(self, epoch, logs=None):
        print('\n————————————————————————————————————')
        model.load_weights(
            'tmp/weights.{epoch:02d}.hdf5'.format(epoch=epoch+1))
        r = model.predict(x_test, verbose=1)
        with open('test.txt', 'w') as f:
            f.write('{}\n'.format(str(r)))
        print('\n————————————————————————————————————')


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = prepare_data(
        r'D:\Workspace\githome\CNN_keras\samples_debug'
    )
    model = build_model()

    print('... compile models')
    model.compile(
        optimizer='adadelta',
        loss=['categorical_crossentropy'] * 5,
        metrics=['accuracy'] * 5,
        loss_weights=[1] * 5,
    )

    check_point = ModelCheckpoint(
        filepath="tmp/weights.{epoch:02d}.hdf5"
    )

    print('... begin train')
    model.fit(
        x_train, y_train, batch_size=128, epochs=100,
        validation_split=0.1, callbacks=[check_point, TestAcc()],
    )
