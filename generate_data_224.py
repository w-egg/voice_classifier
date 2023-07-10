from PIL import Image
import os
import glob
import numpy as np
from sklearn import model_selection

import my_lib

classes = []
image_size = 224


def create_npy():
    X = []
    Y = []

    for index, classlabel in enumerate(classes):
        photos_dir = './face/' + classlabel
        files = glob.glob(photos_dir + '/*.png')
        for i, file in enumerate(files):
            image = Image.open(file)
            image = image.convert("RGB")
            image = image.resize((image_size, image_size))
            data = np.asarray(image)
            X.append(data)
            Y.append(index)
    X = np.array(X)
    Y = np.array(Y)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y)
    xy = (X_train, X_test, y_train, y_test)
    np.save("./imagefiles_224.npy", xy)


if __name__ == "__main__":
    classes = my_lib.add_directory('./face')
    create_npy()
