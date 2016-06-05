import numpy as np
from PIL import Image


def asc_chr(num):
    num = int(num)
    if num < 8:
        answer = chr(num+65)
    elif num == 8:
        answer = 'K'
    elif num == 11:
        answer = 'P'
    elif num == 12:
        answer = 'R'
    elif num > 12:
        answer = chr(num+71)
    else:
        answer = chr(num+68)
    return answer


def asc_num(char):
    if char < 'I':
        answer = ord(char)-65
    elif char == 'K':
        answer = 8
    elif char == 'P':
        answer = 11
    elif char == 'R':
        answer = 12
    elif char > 'S':
        answer = ord(char) - 71
    else:
        answer = ord(char) - 68
    return answer


def load_single_data():
    kv_dict = {}
    with open('label.csv') as f:
        for line in f:
            line = line.strip().split(',')
            key = line[0]
            if key == 'name':
                continue
            pre = key.split('.')[0]
            value = line[1]
            for i, v in enumerate(value):
                kv_dict['%s-%d.jpg' % (pre, i)] = v
    folder = 'sample_single'
    imgs = kv_dict.keys()
    length = len(imgs)
    data = np.empty((length, 3, 40, 40), dtype="float32")
    label = np.empty(length)
    for index, img_name in enumerate(imgs):
        img = Image.open('%s/%s' % (folder, img_name))
        arr = np.asarray(img, dtype="float32")/255.0
        data[index, :, :, :] = np.rollaxis(arr, 2)
        label[index] = asc_num(kv_dict[img_name])
    return data, label
