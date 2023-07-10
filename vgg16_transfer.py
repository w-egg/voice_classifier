import numpy as np
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, adam
from keras.utils import np_utils
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator

import my_lib

classes = []
image_size = 224


if __name__ == '__main__':
    classes = my_lib.add_directory('./face/train')
    num_classes = len(classes)
    print(num_classes)
    #X_train, X_test, y_train, y_test = np.load('./imagefiles_224.npy', allow_pickle=True)
    #y_train = np_utils.to_categorical(y_train, num_classes)
    #y_test = np_utils.to_categorical(y_test, num_classes)
    #x_train = X_train.astype("float") / 255.0
    #X_test = X_test.astype("float") / 255.0

    model = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))

    top_model = model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    predictions = Dense(num_classes, activation='softmax')(top_model)
    model = Model(inputs=model.input, outputs=predictions)

    #top_model = Sequential()
    #top_model.add(Flatten(input_shape=model.output_shape[1:]))
    #top_model.add(Dense(256, activation='relu'))
    #top_model.add(Dropout(0.5))
    #top_model.add(Dense(num_classes, activation='softmax'))

    #model = Model(inputs=model.input, outputs=top_model(model.output))

    for layer in model.layers[:15]:
        layer.trainable = False

    opt = SGD(lr=0.0001, momentum=0.9)
    #opt = adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.fit(X_train, y_train, batch_size=32, epochs=5)
    #score = model.evaluate(X_test, y_test, batch_size=32)
    #print(score)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=10
    )

    test_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
    )

    train_generator = train_datagen.flow_from_directory(
        'face/train',
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = test_datagen.flow_from_directory(
        'face/validation',
        target_size=(image_size, image_size),
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=1600//32,
        epochs=50,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=400//32)
        
    model.save('fruits.hdf5')