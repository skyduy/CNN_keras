import os
import cv2
import numpy as np

APPEARED_LETTERS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'U', 'V', 'W',
    'X', 'Y', 'Z'
]
CAT2CHR = dict(zip(range(len(APPEARED_LETTERS)), APPEARED_LETTERS))
CHR2CAT = dict(zip(APPEARED_LETTERS, range(len(APPEARED_LETTERS))))


def distinct_char(folder):
    chars = set()
    for fn in os.listdir(folder):
        if fn.endswith('.jpg'):
            for letter in fn.split('.')[0]:
                chars.add(letter)
    return sorted(list(chars))


def load_data(folder):
    img_list = [i for i in os.listdir(folder) if i.endswith('jpg')]
    img_nums = len(img_list)
    print('total imgs:', img_nums)
    data = np.empty((img_nums, 40, 150, 3), dtype="float32")  # channel last
    label = np.empty((img_nums, 5))
    for index, img_name in enumerate(img_list):
        img_arr = cv2.imread('{}/{}'.format(folder, img_name)) / 255
        data[index, :, :, :] = img_arr
        label[index] = [CHR2CAT[i] for i in img_name.split('.')[0]]
        if index % 100 == 0:
            print('{} images loads'.format(index))
    return data, label


if __name__ == '__main__':
    print(distinct_char('samples'))
