# Place all early stopping callbacks here

from tensorflow.keras.callbacks import EarlyStopping


early_stopping = {

    "1": EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.0005,
        patience=11,
        verbose=1,
        restore_best_weights=True,
    ),

    "2": EarlyStopping(
        monitor='accuracy',
        min_delta=0.0005,
        patience=15,
        verbose=1,
        restore_best_weights=True,
    ),

    "3": EarlyStopping(
        monitor='accuracy',
        min_delta=0.0006,
        patience=12,
        verbose=1,
        restore_best_weights=True,
    ),

    "4": EarlyStopping(
        monitor='accuracy',
        min_delta=0.0005,
        patience=9,
        verbose=1,
        restore_best_weights=True,
    ),

    "5": EarlyStopping(
        monitor='accuracy',
        min_delta=0.0005,
        patience=7,
        verbose=1,
        restore_best_weights=True,
    ),

}