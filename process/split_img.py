from PIL import Image
from pylab import *


def get_single():
    with open('../label.csv') as label_file:
        for line in label_file:
            pic_name = line.strip().split(',')[0]
            if pic_name == 'name':
                continue
            results = {}
            try:
                raw_img = Image.open('../sample_all/%s' % pic_name)
            except IOError:
                continue
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
                            end = i-1
            centers = [((2*i+1)*end+(9-2*i)*begin)/10 for i in range(5)]
            pre = pic_name.split('.')[0]
            for index, center in enumerate(centers):
                a = center - 19
                b = center + 21
                raw_img.crop((a, 0, b, y_size)).save(
                    '../sample_single/%s-%d.jpg' % (pre, index)
                )

get_single()
