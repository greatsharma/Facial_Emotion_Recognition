# Place all train datagenerators here

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def datagen_img_roi1_roi2(Xtrain_img, Xtrain_roi1, Xtrain_roi2, ytrain, batch_size):
    while True:
        idx = np.random.permutation(Xtrain_img.shape[0])

        datagen = ImageDataGenerator(
            rotation_range=8,
            width_shift_range=0.08,
            height_shift_range=0.08,
            shear_range=0.1,
            zoom_range=0.08,
            horizontal_flip=True,
        )

        batches = datagen.flow(Xtrain_img[idx], ytrain[idx], batch_size=batch_size, shuffle=False)

        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]

            yield [batch[0], Xtrain_roi1[idx[idx0:idx1]], Xtrain_roi2[idx[idx0:idx1]], ], batch[1]

            idx0 = idx1
            if idx1 >= Xtrain_img.shape[0]:
                break


def datagen_img_roi1_roi2_hogfeat(Xtrain_img, Xtrain_roi1, Xtrain_roi2, Xtrain_hogfeat, ytrain, batch_size):
    while True:
        idx = np.random.permutation(Xtrain_img.shape[0])

        datagen = ImageDataGenerator(
            rotation_range=8,
            width_shift_range=0.08,
            height_shift_range=0.08,
            shear_range=0.1,
            zoom_range=0.08,
            horizontal_flip=True,
        )

        batches = datagen.flow(Xtrain_img[idx], ytrain[idx], batch_size=batch_size, shuffle=False)

        idx0 = 0
        for batch in batches:
            idx1 = idx0 + batch[0].shape[0]

            yield [batch[0], Xtrain_roi1[idx[idx0:idx1]], Xtrain_roi2[idx[idx0:idx1]], Xtrain_hogfeat[idx[idx0:idx1]], ], batch[1]

            idx0 = idx1
            if idx1 >= Xtrain_img.shape[0]:
                break


train_datagen = {

    "1": ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    ),

    "2": ImageDataGenerator(
        rotation_range=8,
        width_shift_range=0.08,
        height_shift_range=0.08,
        shear_range=0.1,
        horizontal_flip=True,
    ),

    "3": ImageDataGenerator(
        rotation_range=5,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
    ),

    "4": datagen_img_roi1_roi2,

    "5": datagen_img_roi1_roi2_hogfeat,

}