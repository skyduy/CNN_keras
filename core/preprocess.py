import os
import sys
import numpy as np
import cv2


def load_img(fn, flags=None):
    if sys.platform.startswith('win'):
        if flags is None:
            flags = cv2.IMREAD_UNCHANGED
        img = cv2.imdecode(
            np.fromfile(fn, dtype=np.uint8), flags)
    else:
        img = cv2.imread(fn, flags)
    return img


def save_img(img, fn):
    status = cv2.imwrite(fn, img)
    if status is False:
        print(img)
        cv2.imshow(fn, img)
        cv2.waitKey(0)


def gen_sub_img(raw_img):
    gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    s, e = 0, gray_img.shape[1]
    value = np.sum(gray_img, axis=0)
    for i, v in enumerate(value):
        if v <= 8300:
            s = i
            break
    for i, v in enumerate(value[::-1]):
        if v <= 8200:
            e = e - i
            break

    centers = [((2 * i + 1) * e + (9 - 2 * i) * s) // 10 for i in range(5)]
    for sub_index, center in enumerate(centers):
        a = center - 19
        b = center + 21
        if a < 0:
            a = 0
            b = a + 40
        if b > 150:
            b = 150
            a = b - 40
        yield raw_img[:, a:b, :]


def split_img(src_folder):
    img_list = [i for i in os.listdir(src_folder) if i.endswith('jpg')]
    for index, img_name in enumerate(img_list):
        raw_img = load_img(os.path.join(src_folder, img_name))
        for sub_index, sub_img in enumerate(gen_sub_img(raw_img)):
            cv2.imshow(img_name[sub_index], sub_img)
            cv2.waitKey(0)


if __name__ == '__main__':
    split_img('../samples')
