import joblib
import argparse
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

import utils
import models
import data_builder
import train_datagen
from callbacks import lr_schedulers, early_stopping


print("tensorflow ", tf.__version__, "\n")

ap = argparse.ArgumentParser()

ap.add_argument('-d', '--dataset', required=True,
                help='Dataset to train on, `fer`, `feraligned`, `ck` and `feraligned+ck` are supported')
ap.add_argument('-m', '--model', required=True,
                help='Model to train on, currently `CNNModel`, `CNN_ROI1_ROI2Model` and `CNN_ROI1_ROI2_HOGFeat_Model` are supported')
ap.add_argument('-em', '--emotions', required=True,
                help='Emotions to train on, comma separated values, depending on the dataset select any subset from {Happy,Sadness,Surprise,Angry,Fear,Neutral}')
ap.add_argument('-s', '--shuffle', required=False,
                help="1 to Shuffle before split otherwise 0, default is 1")
ap.add_argument('-rs', '--random_state', required=False,
                help="Random state to use, default is 42")
ap.add_argument('-tr', '--train_ratio', required=False,
                help="Train ratio a value from 0 to 1, default is 0.85")
ap.add_argument('-lrs', '--lr_scheduler', required=False,
                help="lr scheduler to use, default is None")
ap.add_argument('-es', '--early_stopping', required=False,
                help="early stopping to use, default is None")
ap.add_argument('-tg', '--train_datagen', required=False,
                help="train data generator to use, default is None")
ap.add_argument('-bs', '--batch_size', required=False,
                help="Batch size to use, default is 24")
ap.add_argument('-ep', '--epochs', required=False,
                help="Max epochs, default is 50")
ap.add_argument('-o', '--optim', required=False,
                help="Optimizer to use, `adam` and `nadam` are supported, default is adam")
ap.add_argument('-lr', '--learning_rate', required=False,
                help="learning rate to use, default is 0.01")
ap.add_argument('-sa', '--save_architecture', required=False,
                help="1 to save_architecture otherwise 0, default is 0")
ap.add_argument('-sm', '--save_model', required=False,
                help="1 to save the model otherwise 0, default is 0")
ap.add_argument('-scm', '--save_confusion_matrix', required=False,
                help="1 to save the confusion matrix of test set otherwise 0, default is 0")
ap.add_argument('-sth', '--save_training_history', required=False,
                help="1 to save training history otherwise 0, default is 0")


args = vars(ap.parse_args())

DEFAULT_BOOLEAN_PARAMS = {
    'shuffle': True,
    'save_model': False,
    'save_architecture': False,
    'save_confusion_matrix': False,
    'save_training_history': False,
}

for k in args:
    if k in DEFAULT_BOOLEAN_PARAMS:
        args[k] = (
            DEFAULT_BOOLEAN_PARAMS[k]
            if args[k] is None else
            utils.arg2bool(args[k])
        )

DEFAULT_NONBOOLEAN_PARAMS = {
    'random_state': (42, int),
    'train_ratio': (0.85, float),
    'lr_scheduler': (None, str),
    'early_stopping': (None, str),
    'train_generator': (None, str),
    'batch_size': (24, int),
    'epochs': (50, int),
    'optim': ("adam", str),
    'learning_rate': (0.01, float),
}

for k in args:
    if k in DEFAULT_NONBOOLEAN_PARAMS:
        args[k] = (
            DEFAULT_NONBOOLEAN_PARAMS[k][0]
            if args[k] is None else
            DEFAULT_NONBOOLEAN_PARAMS[k][1](args[k])
        )

DATA_PATH = "inputs/" + args["dataset"] + "/"
OUTPUT_PATH = "outputs/"
EMOTIONS = list(args["emotions"].split(","))

callbacks = []
if not args["lr_scheduler"] is None:
    callbacks.append(lr_schedulers.lr_schedulers[args["lr_scheduler"]])
if not args["early_stopping"] is None:
    callbacks.append(early_stopping.early_stopping[args["early_stopping"]])

if not args["train_datagen"] is None:
    train_datagen = train_datagen.train_datagen[args["train_datagen"]]
else:
    train_datagen = None

if args["optim"] == "nadam":
    optim = optimizers.Nadam(args["learning_rate"])
else:
    optim = optimizers.Adam(args["learning_rate"])


if args["model"] == "CNNModel":
    model = models.CNNModel()
    img_arr, img_label, label_to_text = data_builder.ImageToArray(DATA_PATH, EMOTIONS).build_from_directory()
    img_arr = img_arr / 255.

    X_train, X_test, y_train, y_test = train_test_split(img_arr, img_label, shuffle=args["shuffle"], stratify=img_label,
                                                        train_size=args["train_ratio"], random_state=args["random_state"])
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape} \n")
    
    model.train(
        X_train, y_train,
        validation_data = (X_test, y_test),
        batch_size = args["batch_size"],
        epochs = args["epochs"],
        optim = optim,
        callbacks = callbacks,
        train_datagen = train_datagen,
    )

    RUN_NAME = f"{model.__class__.__name__}_{args['dataset']}_{len(EMOTIONS)}emo"

    if args["save_confusion_matrix"]:
        model.evaluate(X_test, y_test, OUTPUT_PATH + "confusion_matrix/" + RUN_NAME + ".png")

elif args["model"] == "CNN_ROI1_ROI2Model":
    model = models.CNN_ROI1_ROI2Model()
    roi1_arr, roi2_arr, img_to_exclude = data_builder.ImageToROI(DATA_PATH, EMOTIONS).build_from_directory()

    img2arr_obj = data_builder.ImageToArray(DATA_PATH, EMOTIONS, img_to_exclude)
    img_arr, img_label, label_to_text = img2arr_obj.build_from_directory()
    img2arr_obj.class_image_count()

    img_arr = img_arr / 255.
    roi1_arr = roi1_arr / 255.
    roi2_arr = roi2_arr / 255.

    Xtrain_img, Xtest_img, Xtrain_roi1, Xtest_roi1, Xtrain_roi2, Xtest_roi2, y_train, y_test =\
    train_test_split(img_arr, roi1_arr, roi2_arr, img_label,
                    shuffle=args["shuffle"], stratify=img_label, train_size=args["train_ratio"], random_state=args["random_state"])

    print(f"Xtrain_img: {Xtrain_img.shape}, Xtrain_roi1: {Xtrain_roi1.shape}, Xtrain_roi2: {Xtrain_roi2.shape}, y_train: {y_train.shape}")
    print(f"Xtest_img: {Xtest_img.shape}, Xtest_roi1: {Xtest_roi1.shape}, Xtest_roi2: {Xtest_roi2.shape}, y_test: {y_test.shape} \n")

    model.train(
        Xtrain_img, Xtrain_roi1, Xtrain_roi2,
        y_train,
        validation_data = ([Xtest_img, Xtest_roi1, Xtest_roi2], y_test),
        batch_size = args["batch_size"],
        epochs = args["epochs"],
        optim = optim,
        callbacks = callbacks,
        train_datagen = train_datagen,
    )

    RUN_NAME = f"{model.__class__.__name__}_{args['dataset']}_{len(EMOTIONS)}emo"

    if args["save_confusion_matrix"]:
        model.evaluate([Xtest_img, Xtest_roi1, Xtest_roi2], y_test, OUTPUT_PATH + "confusion_matrix/" + RUN_NAME + ".png")

elif args["model"] == "CNN_ROI1_ROI2_HOGFeat_Model":
    model = models.CNN_ROI1_ROI2_HOGFeat_Model()
    
    roi1_arr, roi2_arr, img_to_exclude = data_builder.ImageToROI(DATA_PATH, EMOTIONS).build_from_directory()
    hogfeat = data_builder.ImageToHOGFeatures(DATA_PATH, EMOTIONS, img_to_exclude).build_from_directory()

    img2arr_obj = data_builder.ImageToArray(DATA_PATH, EMOTIONS, img_to_exclude)
    img_arr, img_label, label_to_text = img2arr_obj.build_from_directory()
    img2arr_obj.class_image_count()

    img_arr = img_arr / 255.
    roi1_arr = roi1_arr / 255.
    roi2_arr = roi2_arr / 255.

    Xtrain_img, Xtest_img, Xtrain_roi1, Xtest_roi1, Xtrain_roi2, Xtest_roi2, Xtrain_hogfeat, Xtest_hogfeat, y_train, y_test =\
    train_test_split(img_arr, roi1_arr, roi2_arr, hogfeat, img_label,
                    shuffle=args["shuffle"], stratify=img_label, train_size=args["train_ratio"], random_state=args["random_state"])

    print(f"Xtrain_img: {Xtrain_img.shape}, Xtrain_roi1: {Xtrain_roi1.shape}, Xtrain_roi2: {Xtrain_roi2.shape}, Xtrain_hogfeat: {Xtrain_hogfeat.shape}, y_train: {y_train.shape}")
    print(f"Xtest_img: {Xtest_img.shape}, Xtest_roi1: {Xtest_roi1.shape}, Xtest_roi2: {Xtest_roi2.shape}, Xtest_hogfeat: {Xtest_hogfeat.shape}, y_test: {y_test.shape} \n")

    model.train(
        Xtrain_img, Xtrain_roi1, Xtrain_roi2, Xtrain_hogfeat,
        y_train,
        validation_data = ([Xtest_img, Xtest_roi1, Xtest_roi2, Xtest_hogfeat], y_test),
        batch_size = args["batch_size"],
        epochs = args["epochs"],
        optim = optim,
        callbacks = callbacks,
        train_datagen = train_datagen,
    )

    RUN_NAME = f"{model.__class__.__name__}_{args['dataset']}_{len(EMOTIONS)}emo"

    if args["save_confusion_matrix"]:
        model.evaluate([Xtest_img, Xtest_roi1, Xtest_roi2, Xtest_hogfeat], y_test, OUTPUT_PATH + "confusion_matrix/" + RUN_NAME + ".png")

elif args["model"] == "CNN_ROI1_ROI2_KLDIST_Model":
    model = models.CNN_ROI1_ROI2_KLDIST_Model()
    
    roi1_arr, roi2_arr, img_to_exclude = data_builder.ImageToROI(DATA_PATH, EMOTIONS).build_from_directory()
    kl_dists = data_builder.ImageToKeyLandmarksDistances(DATA_PATH, EMOTIONS, img_to_exclude).build_from_directory()

    img2arr_obj = data_builder.ImageToArray(DATA_PATH, EMOTIONS, img_to_exclude)
    img_arr, img_label, label_to_text = img2arr_obj.build_from_directory()
    img2arr_obj.class_image_count()

    img_arr = img_arr / 255.
    roi1_arr = roi1_arr / 255.
    roi2_arr = roi2_arr / 255.

    Xtrain_img, Xtest_img, Xtrain_roi1, Xtest_roi1, Xtrain_roi2, Xtest_roi2, Xtrain_kldist, Xtest_kldist, y_train, y_test =\
    train_test_split(img_arr, roi1_arr, roi2_arr, kl_dists, img_label,
                    shuffle=args["shuffle"], stratify=img_label, train_size=args["train_ratio"], random_state=args["random_state"])

    print(f"Xtrain_img: {Xtrain_img.shape}, Xtrain_roi1: {Xtrain_roi1.shape}, Xtrain_roi2: {Xtrain_roi2.shape}, Xtrain_kldist: {Xtrain_kldist.shape}, y_train: {y_train.shape}")
    print(f"Xtest_img: {Xtest_img.shape}, Xtest_roi1: {Xtest_roi1.shape}, Xtest_roi2: {Xtest_roi2.shape}, Xtest_kldist: {Xtest_kldist.shape}, y_test: {y_test.shape} \n")

    model.train(
        Xtrain_img, Xtrain_roi1, Xtrain_roi2, Xtrain_kldist,
        y_train,
        validation_data = ([Xtest_img, Xtest_roi1, Xtest_roi2, Xtest_kldist], y_test),
        batch_size = args["batch_size"],
        epochs = args["epochs"],
        optim = optim,
        callbacks = callbacks,
        train_datagen = train_datagen,
    )

    RUN_NAME = f"{model.__class__.__name__}_{args['dataset']}_{len(EMOTIONS)}emo"

    if args["save_confusion_matrix"]:
        model.evaluate([Xtest_img, Xtest_roi1, Xtest_roi2, Xtest_kldist], y_test, OUTPUT_PATH + "confusion_matrix/" + RUN_NAME + ".png")

else:
    raise ValueError(f"Invalid model {args['model']}, only `CNNModel`, `CNN_ROI1_ROI2Model` and `CNN_ROI1_ROI2_HOGFeat_Model` are supported")


if args["save_model"]:
    model.save_model(OUTPUT_PATH + "models/" + RUN_NAME + ".h5")
    print(label_to_text)
    joblib.dump(label_to_text, OUTPUT_PATH + "label2text/label2text_" + RUN_NAME + ".pkl")
if args["save_training_history"]:
    model.save_training_history(OUTPUT_PATH + "epoch_metrics/" + RUN_NAME + ".png")
if args["save_architecture"]:
    plot_model(model.model, show_shapes=True, show_layer_names=True, expand_nested=True,
               dpi=50, to_file=OUTPUT_PATH + "architectures/" + model.__class__.__name__ + ".png")