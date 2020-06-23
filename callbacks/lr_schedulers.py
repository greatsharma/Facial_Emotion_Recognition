# Place all lr scheduler callbacks here

from tensorflow.keras.callbacks import ReduceLROnPlateau


lr_schedulers = {

    "1": ReduceLROnPlateau(
        monitor='val_accuracy',
        min_delta=0.0001,
        factor=0.1,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    ),

    "2": ReduceLROnPlateau(
        monitor='val_accuracy',
        min_delta=0.0001,
        factor=0.5,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    ),

    "3": ReduceLROnPlateau(
        monitor='val_accuracy',
        min_delta=0.0001,
        factor=0.2,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    ),

    "4": ReduceLROnPlateau(
        monitor='val_accuracy',
        min_delta=0.0001,
        factor=0.35,
        patience=4,
        min_lr=1e-7,
        verbose=1,
    ),

}