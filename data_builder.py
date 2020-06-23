import os
import cv2
import dlib
import numpy as np
from zipfile import ZipFile
from sklearn.preprocessing import OneHotEncoder

import utils


face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("face_detectors/shape_predictor_68_face_landmarks.dat")


class DataBuilder():

    def __init__(self, path, classes, img_to_exclude=[]):
        self.path = path
        self.classes = classes
        self.img_to_exclude = img_to_exclude

    def class_image_count(self):
        total_images = 0
        for dir_ in os.listdir(self.path):
            if dir_ in self.classes:
                count = 0
                for f in os.listdir(self.path + dir_ + "/"):
                    if not dir_ + "/" + f in self.img_to_exclude:
                        count += 1
                print(f"class {dir_} has {count} images")
                total_images += count
        print(f"total images are {total_images}")

    def build_from_directory(self):
        raise NotImplementedError("build_from_directory is not implemented")

    def build_from_zip(self, path_from, path_to):
        with ZipFile(self.path_from, 'r') as zip_:
            print('Extracting all the files...') 
            zip_.extractall(path_to)
            print('Done!')
        self.zip_extractor(path_to)
        self.path = path_to
        self.build_from_directory()


class ImageToArray(DataBuilder):

    def build_from_directory(self):
        img_arr = []
        img_label = []
        label_to_text = {}
        label = 0

        for dir_ in os.listdir(self.path):
            if dir_ in self.classes:
                for f in os.listdir(self.path + dir_ + "/"):
                    if not dir_ + "/" + f in self.img_to_exclude:
                        img_arr.append(np.expand_dims(cv2.imread(self.path + dir_ + "/" + f, 0), axis=2))
                        img_label.append(label)
                print(f"loaded {dir_} images to numpy arrays...")
                label_to_text[label] = dir_
                label += 1

        img_arr = np.array(img_arr)
        img_label = np.array(img_label)
        img_label = OneHotEncoder(sparse=False).fit_transform(img_label.reshape(-1,1))
        
        return img_arr, img_label, label_to_text


class ImageToROI(DataBuilder):
    
    def build_from_directory(self):
        print("Extracting Eyes and mouth ROI, this may take some time")

        roi1_arr = []
        roi2_arr = []
        self.img_to_exclude = []

        for dir_ in os.listdir(self.path):
            if dir_ in self.classes:
                print(f"processing {dir_} images...")

                for f in os.listdir(self.path + dir_ + "/"):
                    gray_img = cv2.imread(self.path + dir_ + "/" + f, 0)
                    gray_img = cv2.resize(gray_img, (96,96))

                    faces = face_detector(gray_img)
                    if faces:
                        for face in faces:
                            try:
                                landmarks = shape_predictor(gray_img, face)
                                roi1, roi2 = utils.extract_roi1_roi2(gray_img, landmarks)
                                roi1_arr.append(roi1), roi2_arr.append(roi2)
                            except Exception:
                                self.img_to_exclude.append(dir_ + "/" + f)
                                break
                    else:
                        self.img_to_exclude.append(dir_ + "/" + f)

        print(f"\ntotal images to exclude: {len(self.img_to_exclude)}")
        return np.array(roi1_arr), np.array(roi2_arr), self.img_to_exclude


class ImageToHOGFeatures(DataBuilder):

    def build_from_directory(self):
        print("Extracting HOG Features...")

        hog_features = []
        for dir_ in os.listdir(self.path):
            if dir_ in self.classes:
                for f in os.listdir(self.path + dir_ + "/"):
                    if not dir_ + "/" + f in self.img_to_exclude:
                        gray_img = cv2.imread(self.path + dir_ + "/" + f, 0)
                        hogfeat = utils.extract_hog_features(gray_img)
                        hog_features.append(hogfeat)
                print(f"processed {dir_} images...")

        return np.array(hog_features)


class ImageToKeyLandmarksDistances(DataBuilder):

    def build_from_directory(self):
        print("Extracting KeyLandmarks Distances...")

        kl_distances = []
        for dir_ in os.listdir(self.path):
            if dir_ in self.classes:
                print(f"processing {dir_} images...")

                for f in os.listdir(self.path + dir_ + "/"):
                    if not dir_ + "/" + f in self.img_to_exclude:
                        gray_img = cv2.imread(self.path + dir_ + "/" + f, 0)
                        gray_img = cv2.resize(gray_img, (96,96))
                        
                        faces = face_detector(gray_img)
                        landmarks_coord = utils.get_landmarks(gray_img, gray_img, faces[0])
                        key_landmarks_coords = utils.get_keylandmarks_coords(landmarks_coord)

                        all_kl_dists = utils.get_keylandmarks_distances(key_landmarks_coords)
                        kl_distances.append(all_kl_dists)

        return np.array(kl_distances)