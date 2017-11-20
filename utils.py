import os
import numpy as np

from PIL import Image

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
    imgs = [i for i in os.listdir(folder) if i.endswith('jpg')]
    img_nums = len(imgs)
    data = np.empty((img_nums, 3, 150, 40), dtype="float32")
    label = np.empty((img_nums, 5))
    for index, img_name in enumerate(imgs):
        img = Image.open('%s/%s' % (folder, img_name))
        arr = np.asarray(img, dtype="float32")/255.0
        data[index, :, :, :] = np.rollaxis(np.rollaxis(arr, 2), 2, 1)
        label[index, :] = [CHR2CAT[i] for i in img_name.split('.')[0]]
    return data, label


if __name__ == '__main__':
    print(distinct_char('samples'))
