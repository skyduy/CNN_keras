import os
import cv2
import numpy as np
from core import preprocess

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
    letters_num = len(img_list) * 5
    print('total letters:', letters_num)
    data = np.empty((letters_num, 40, 40, 3), dtype="uint8")  # channel last
    label = np.empty((letters_num,))
    for index, img_name in enumerate(img_list):
        raw_img = preprocess.load_img(os.path.join(folder, img_name))
        sub_imgs = preprocess.gen_sub_img(raw_img)
        for sub_index, img in enumerate(sub_imgs):
            data[index*5+sub_index, :, :, :] = img / 255
            label[index*5+sub_index] = CHR2CAT[img_name[sub_index]]
        if index % 100 == 0:
            print('{} letters loads'.format(index*5))
    return data, label


if __name__ == '__main__':
    # print(distinct_char('../data'))
    d, l = load_data('../samples')
    for n, i in enumerate(d):
        cv2.imshow(CAT2CHR[l[n]], i*255)
        print(CAT2CHR[l[n]])
        cv2.waitKey(0)
