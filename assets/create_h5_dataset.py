import h5py as h5
from PIL import Image
import numpy as np
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def read():
    f = h5.File('./train_maskvnomask.h5')
    x = f['train_set_x']
    y = f['train_set_y']
    c = f['list_classes']
    print(x.shape)


def write_x(dir_path, dset, offset=0):
    xs = []
    for file in os.listdir(dir_path):
        xs.append(img_to_arr(dir_path + "\\" + file))
    for i in range(len(xs)):
        dset[i + offset] = xs[i]
    return dset


def write_y(length, y_val, dset, offset=0):
    for i in range(length):
        dset[i + offset] = y_val
    return dset


def img_to_arr(path):
    img = Image.open(path)
    img.load()
    img = img.resize((64, 64))
    return np.asarray(img, dtype='uint8')


def mask_nomask_dataset():
    f = h5.File('test.h5', 'w')
    x = f.create_dataset('x', (1435, 64, 64, 3), dtype='uint8')
    y = f.create_dataset('y', (1435,), dtype='int64')
    x = write_x('./dataset/dataset/with_mask', x)
    x = write_x('./dataset/dataset/without_mask', x, 1000)
    y = write_y(1000, 1, y)
    y = write_y(435, 0, y, 1000)
    print(list(np.asarray(y)))
    f.close()


def train_test_dataset():
    f = h5.File('test.h5', 'r')
    x_orig = np.asarray(f['x'])
    y_orig = np.asarray(f['y'])
    x, y = shuffle(x_orig, y_orig)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.33)
    list_classes = np.array(['no-mask', 'mask'], dtype='|S7')
    train_maskvnomask = h5.File('train_maskvnomask.h5', 'w')
    train_x_d = train_maskvnomask.create_dataset('train_set_x', x_train.shape, dtype='uint8', data=x_train)
    train_y_d = train_maskvnomask.create_dataset('train_set_y', y_train.shape, dtype='int64', data=y_train)
    train_class_d = train_maskvnomask.create_dataset('list_classes', (2,), dtype='|S7', data=list_classes)
    test_maskvnomask = h5.File('test_maskvnomask.h5', 'w')
    test_x_d = test_maskvnomask.create_dataset('test_set_x', x_test.shape, dtype='uint8', data=x_test)
    test_y_d = test_maskvnomask.create_dataset('test_set_y', y_test.shape, dtype='int64', data=y_test)
    test_class_d = test_maskvnomask.create_dataset('list_classes', (2,), dtype='|S7', data=list_classes)
    train_maskvnomask.close()
    test_maskvnomask.close()


def main():
    read()


if __name__ == '__main__':
    main()
