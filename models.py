import numpy as np

import scikitplot
import seaborn as sns
from matplotlib import pyplot

from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Concatenate
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.utils import plot_model


class BaseModel():

    def __init__(self):
        self.model = None
        self.trained = False

    def model_builder(self, in_shape, out_shape, optim, loss, **kwargs):
        raise NotImplementedError("model_builder is not implemented")

    def train(self):
        raise NotImplementedError("train is not implemented")

    def evaluate(self, X_test, y_test, save_evaluation_to=None):
        if self.trained:
            yhat_test = np.argmax(self.model.predict(X_test), axis=1)
            ytest_ = np.argmax(y_test, axis=1)
            test_accu = np.sum(ytest_ == yhat_test) / len(ytest_) * 100
            print(f"test accuracy: {round(test_accu, 4)} %\n\n")
            print(classification_report(ytest_, yhat_test))

            if not save_evaluation_to is None:
                scikitplot.metrics.plot_confusion_matrix(ytest_, yhat_test, figsize=(7,7))
                pyplot.savefig(save_evaluation_to)
        else:
            raise ValueError("Model is not trained yet, call train first")
            
    def predict(self, X, classes=True):
        return (
            np.argmax(self.model.predict(X), axis=1)
            if classes else
            self.model.predict(X)
        )

    def save_model(self, path):
        if self.trained:
            self.model.save(path)
        else:
            raise ValueError("Model is not trained yet, call train first")

    def save_training_history(self, path):
        if self.trained:
            sns.set()
            fig = pyplot.figure(0, (12, 4))

            ax = pyplot.subplot(1, 2, 1)
            sns.lineplot(self.history.epoch, self.history.history['accuracy'], label='train')
            try:
                sns.lineplot(self.history.epoch, self.history.history['val_accuracy'], label='valid')
            except KeyError:
                pass
            pyplot.title('Accuracy')
            pyplot.tight_layout()

            ax = pyplot.subplot(1, 2, 2)
            sns.lineplot(self.history.epoch, self.history.history['loss'], label='train')
            try:
                sns.lineplot(self.history.epoch, self.history.history['val_loss'], label='valid')
            except KeyError:
                pass
            pyplot.title('Loss')
            pyplot.tight_layout()
            pyplot.savefig(path)
        else:
            raise ValueError("Model is not trained yet, call train first")

    def __repr__(self):
        return f"model: {self.__class__.__name__},  trained: {self.trained}"


def cnn_for_raw_img(in_shape, out_shape):
    model_in = Input(shape=in_shape, name="input_CNN")
    
    conv2d_1 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_1'
    )(model_in)
    batchnorm_1 = BatchNormalization(name='batchnorm_1')(conv2d_1)
    conv2d_2 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_2'
    )(batchnorm_1)
    batchnorm_2 = BatchNormalization(name='batchnorm_2')(conv2d_2)
    
    maxpool2d_1 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_1')(batchnorm_2)
    dropout_1 = Dropout(0.35, name='dropout_1')(maxpool2d_1)

    conv2d_3 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_3'
    )(dropout_1)
    batchnorm_3 = BatchNormalization(name='batchnorm_3')(conv2d_3)
    conv2d_4 = Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_4'
    )(batchnorm_3)
    batchnorm_4 = BatchNormalization(name='batchnorm_4')(conv2d_4)
    
    maxpool2d_2 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_2')(batchnorm_4)
    dropout_2 = Dropout(0.4, name='dropout_2')(maxpool2d_2)

    conv2d_5 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_5'
    )(dropout_2)
    batchnorm_5 = BatchNormalization(name='batchnorm_5')(conv2d_5)
    conv2d_6 = Conv2D(
        filters=256,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_6'
    )(batchnorm_5)
    batchnorm_6 = BatchNormalization(name='batchnorm_6')(conv2d_6)
    
    maxpool2d_3 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_3')(batchnorm_6)
    dropout_3 = Dropout(0.5, name='dropout_3')(maxpool2d_3)

    flatten = Flatten(name='flatten')(dropout_3)
        
    dense_1 = Dense(
        256,
        activation='elu',
        kernel_initializer='he_normal',
        name='dense1'
    )(flatten)
    batchnorm_7 = BatchNormalization(name='batchnorm_7')(dense_1)

    model_out = Dropout(0.6, name='dropout_4')(batchnorm_7)
    return model_in, model_out


class CNNModel(BaseModel):

    def model_builder(self, in_shape, out_shape):
        cnn_in, cnn_out = cnn_for_raw_img(in_shape, out_shape)
        model_out = Dense(out_shape, activation="softmax", name="out_layer")(cnn_out)
        self.model = Model(inputs=cnn_in, outputs=model_out, name="CNN")

    def train(self, X_train, y_train, validation_data, batch_size=24, epochs=50,
              optim=optimizers.Adam(0.01), callbacks=[], train_datagen=None):

        self.model_builder(X_train.shape[1:], y_train.shape[1])

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optim,
            metrics=['accuracy']
        )

        if train_datagen is None:
            self.history = self.model.fit(
                X_train, y_train,
                validation_data = validation_data,
                batch_size = batch_size,
                epochs = epochs,
                callbacks = callbacks,
            )
        else:
            steps_per_epoch = len(X_train) / batch_size
            self.history = self.model.fit(
                train_datagen.flow(X_train, y_train, batch_size=batch_size),
                validation_data = validation_data,
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                callbacks = callbacks,
            )

        self.trained = True


def cnn_for_roi1_img(in_shape):
    model_in = Input(shape=in_shape, name="input_ROI1")
    
    conv2d_1 = Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_1_roi1'
    )(model_in)
    batchnorm_1 = BatchNormalization(name='batchnorm_1_roi1')(conv2d_1)
    conv2d_2 = Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_2_roi1'
    )(batchnorm_1)
    batchnorm_2 = BatchNormalization(name='batchnorm_2_roi1')(conv2d_2)
    
    maxpool2d_1 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_1_roi1')(batchnorm_2)
    dropout_1 = Dropout(0.4, name='dropout_1_roi1')(maxpool2d_1)

    conv2d_3 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_3_roi1'
    )(dropout_1)
    batchnorm_3 = BatchNormalization(name='batchnorm_3_roi1')(conv2d_3)
    conv2d_4 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_4_roi1'
    )(batchnorm_3)
    batchnorm_4 = BatchNormalization(name='batchnorm_4_roi1')(conv2d_4)
    
    maxpool2d_2 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_2_roi1')(batchnorm_4)
    dropout_2 = Dropout(0.4, name='dropout_2_roi1')(maxpool2d_2)

    flatten = Flatten(name='flatten_roi1')(dropout_2)
        
    dense_1 = Dense(
        128,
        activation='elu',
        kernel_initializer='he_normal',
        name='dense1_roi1'
    )(flatten)
    batchnorm_5 = BatchNormalization(name='batchnorm_5_roi1')(dense_1)
    
    model_out = Dropout(0.6, name='dropout_3_roi1')(batchnorm_5)
    return model_in, model_out


def cnn_for_roi2_img(in_shape):
    model_in = Input(shape=in_shape, name="input_ROI2")
    
    conv2d_1 = Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_1_roi2'
    )(model_in)
    batchnorm_1 = BatchNormalization(name='batchnorm_1_roi2')(conv2d_1)
    conv2d_2 = Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_2_roi2'
    )(batchnorm_1)
    batchnorm_2 = BatchNormalization(name='batchnorm_2_roi2')(conv2d_2)
    
    maxpool2d_1 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_1_roi2')(batchnorm_2)
    dropout_1 = Dropout(0.4, name='dropout_1_roi2')(maxpool2d_1)

    conv2d_3 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_3_roi2'
    )(dropout_1)
    batchnorm_3 = BatchNormalization(name='batchnorm_3_roi2')(conv2d_3)
    conv2d_4 = Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation='elu',
        padding='same',
        kernel_initializer='he_normal',
        name='conv2d_4_roi2'
    )(batchnorm_3)
    batchnorm_4 = BatchNormalization(name='batchnorm_4_roi2')(conv2d_4)
    
    maxpool2d_2 = MaxPooling2D(pool_size=(2,2), name='maxpool2d_2_roi2')(batchnorm_4)
    dropout_2 = Dropout(0.4, name='dropout_2_roi2')(maxpool2d_2)

    flatten = Flatten(name='flatten_roi2')(dropout_2)
        
    dense_1 = Dense(
        128,
        activation='elu',
        kernel_initializer='he_normal',
        name='dense1_roi2'
    )(flatten)
    batchnorm_5 = BatchNormalization(name='batchnorm_5_roi2')(dense_1)

    model_out = Dropout(0.6, name='dropout_3_roi2')(batchnorm_5)
    return model_in, model_out


class CNN_ROI1_ROI2Model(BaseModel):

    def model_builder(self, in_shape, out_shape):
        rawimg_in, rawimg_out = cnn_for_raw_img(in_shape=in_shape, out_shape=out_shape)
        roi1_in, roi1_out = cnn_for_roi1_img(in_shape=(25,50,1))
        roi2_in, roi2_out = cnn_for_roi2_img(in_shape=(25,50,1))

        models_in=[rawimg_in, roi1_in, roi2_in,],
        models_out=[rawimg_out, roi1_out, roi2_out,]

        concated = Concatenate()(models_out)
        dropout_1 = Dropout(0.3, name='dropout1_merged')(concated)

        dense1 = Dense(128, activation="elu", name="dense1_merged")(dropout_1)
        dropout_2 = Dropout(0.4, name='dropout2_merged')(dense1)

        out = Dense(out_shape, activation="softmax", name="out_layer")(dropout_2)
        self.model = Model(inputs=models_in, outputs=out, name="CNN_ROI1_ROI2")

    def train(self, Xtrain_img, Xtrain_roi1, Xtrain_roi2, y_train, validation_data, batch_size=24, epochs=50,
              optim=optimizers.Adam(0.01), callbacks=[], train_datagen=None):

        self.model_builder(Xtrain_img.shape[1:], y_train.shape[1])

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optim,
            metrics=['accuracy']
        )

        if train_datagen is None:
            self.history = self.model.fit(
                x = [Xtrain_img, Xtrain_roi1, Xtrain_roi2], 
                y = y_train,
                validation_data = validation_data,
                batch_size = batch_size,
                epochs = epochs,
                callbacks = callbacks,
            )
        else:
            steps_per_epoch = len(Xtrain_img) / batch_size
            self.history = self.model.fit(
                train_datagen(Xtrain_img, Xtrain_roi1, Xtrain_roi2, y_train, batch_size=batch_size),
                validation_data = validation_data,
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                callbacks = callbacks,
            )

        self.trained = True


def nn_for_hogfeat(in_shape):
    model_in = Input(shape=in_shape, name="input_HOGfeat")
    flatten = Flatten(name="flatten_hogfeat")(model_in)
    dense1 = Dense(512, activation="elu", name="dense1_hogfeat")(flatten)
    model_out = Dropout(0.4, name='dropout1_hogfeat')(dense1)
    return model_in, model_out


class CNN_ROI1_ROI2_HOGFeat_Model(BaseModel):

    def model_builder(self, in_shape, out_shape):
        rawimg_in, rawimg_out = cnn_for_raw_img(in_shape=in_shape, out_shape=out_shape)
        roi1_in, roi1_out = cnn_for_roi1_img(in_shape=(25,50,1))
        roi2_in, roi2_out = cnn_for_roi2_img(in_shape=(25,50,1))
        hogfeat_in, hogfeat_out = nn_for_hogfeat(in_shape=(3780,1))

        models_in=[rawimg_in, roi1_in, roi2_in, hogfeat_in],
        models_out=[rawimg_out, roi1_out, roi2_out, hogfeat_out]

        concated = Concatenate()(models_out)
        dropout_1 = Dropout(0.3, name='dropout1_merged')(concated)

        dense1 = Dense(256, activation="elu", name="dense1_merged")(dropout_1)
        dropout_2 = Dropout(0.4, name='dropout2_merged')(dense1)

        dense2 = Dense(128, activation="elu", name="dense2_merged")(dropout_2)
        dropout_3 = Dropout(0.4, name='dropout3_merged')(dense2)

        out = Dense(out_shape, activation="softmax", name="out_layer")(dropout_3)
        self.model = Model(inputs=models_in, outputs=out, name="CNN_ROI1_ROI2_HOGFEAT")

    def train(self, Xtrain_img, Xtrain_roi1, Xtrain_roi2, Xtrain_hogfeat, y_train, validation_data, batch_size=24, epochs=50,
              optim=optimizers.Adam(0.01), callbacks=[], train_datagen=None):

        self.model_builder(Xtrain_img.shape[1:], y_train.shape[1])

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optim,
            metrics=['accuracy']
        )

        if train_datagen is None:
            self.history = self.model.fit(
                x = [Xtrain_img, Xtrain_roi1, Xtrain_roi2, Xtrain_hogfeat], 
                y = y_train,
                validation_data = validation_data,
                batch_size = batch_size,
                epochs = epochs,
                callbacks = callbacks,
            )
        else:
            steps_per_epoch = len(Xtrain_img) / batch_size
            self.history = self.model.fit(
                train_datagen(Xtrain_img, Xtrain_roi1, Xtrain_roi2, Xtrain_hogfeat, y_train, batch_size=batch_size),
                validation_data = validation_data,
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                callbacks = callbacks,
            )

        self.trained = True


def nn_for_keylandmarks_distance(input_shape):
    model_in = Input(shape=input_shape, name="input_KLDist")
    flatten = Flatten(name="flatten_kldist")(model_in)
    dense1 = Dense(32, activation="elu", name="dense1_kldist")(flatten)
    model_out = Dropout(0.3, name='dropout1_kldist')(dense1)
    return model_in, model_out


class CNN_ROI1_ROI2_KLDIST_Model(BaseModel):

    def model_builder(self, in_shape, out_shape):
        rawimg_in, rawimg_out = cnn_for_raw_img(in_shape=in_shape, out_shape=out_shape)
        roi1_in, roi1_out = cnn_for_roi1_img(in_shape=(25,50,1))
        roi2_in, roi2_out = cnn_for_roi2_img(in_shape=(25,50,1))
        kldist_in, kldist_out = nn_for_keylandmarks_distance(input_shape=(77,1))
        
        models_in=[rawimg_in, roi1_in, roi2_in, kldist_in],
        models_out=[rawimg_out, roi1_out, roi2_out, kldist_out]

        concated = Concatenate()(models_out)
        dropout_1 = Dropout(0.3, name='dropout1_merged')(concated)

        dense1 = Dense(256, activation="elu", name="dense1_merged")(dropout_1)
        dropout_2 = Dropout(0.4, name='dropout2_merged')(dense1)

        dense2 = Dense(128, activation="elu", name="dense2_merged")(dropout_2)
        dropout_3 = Dropout(0.4, name='dropout3_merged')(dense2)

        out = Dense(out_shape, activation="softmax", name="out_layer")(dropout_3)
        self.model = Model(inputs=models_in, outputs=out, name="CNN_ROI1_ROI2_HOGFEAT")

    def train(self, Xtrain_img, Xtrain_roi1, Xtrain_roi2, Xtrain_kldist, y_train, validation_data, batch_size=24, epochs=50,
              optim=optimizers.Adam(0.01), callbacks=[], train_datagen=None):

        self.model_builder(Xtrain_img.shape[1:], y_train.shape[1])

        self.model.compile(
            loss="categorical_crossentropy",
            optimizer=optim,
            metrics=['accuracy']
        )

        if train_datagen is None:
            self.history = self.model.fit(
                x = [Xtrain_img, Xtrain_roi1, Xtrain_roi2, Xtrain_kldist], 
                y = y_train,
                validation_data = validation_data,
                batch_size = batch_size,
                epochs = epochs,
                callbacks = callbacks,
            )
        else:
            steps_per_epoch = len(Xtrain_img) / batch_size
            self.history = self.model.fit(
                train_datagen(Xtrain_img, Xtrain_roi1, Xtrain_roi2, Xtrain_kldist, y_train, batch_size=batch_size),
                validation_data = validation_data,
                steps_per_epoch = steps_per_epoch,
                epochs = epochs,
                callbacks = callbacks,
            )

        self.trained = True