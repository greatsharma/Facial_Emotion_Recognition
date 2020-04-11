import warnings
warnings.filterwarnings('ignore')

import os

MODEL_PATH = os.environ.get("MODEL_PATH")
TEST_SAMPLE_PATH = os.environ.get("TEST_SAMPLE_PATH")
SAMPLE_TYPE = os.environ.get("SAMPLE_TYPE")
MODEL = os.environ.get("MODEL")

if not SAMPLE_TYPE in ["image", "video", "webcam"]:
    raise ValueError(f"sample type should be `image`, `video` or `webcam` but got {SAMPLE_TYPE}")

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras import preprocessing


cascade_lassifiers = [
    "haarcascade_frontalface",
    "lbpcascade_frontalface",
    "lbpcascade_profileface",
]

haarcascade = cv2.CascadeClassifier(MODEL_PATH + f"{cascade_lassifiers[1]}.xml")
model = load_model(MODEL_PATH + MODEL)

emotions = ['happy', 'sad', 'neutral']


def preprocess_img(img):
    img = cv2.resize(img, (48,48))
    img_arr = preprocessing.image.img_to_array(img)
    img_arr = img_arr.astype("float32")
    img_arr = img_arr / 255.
    img_arr = np.expand_dims(img_arr, axis = 0)
    return img_arr


def get_emotion(img_arr):
    predicted_label = np.argmax(model.predict(img_arr)[0])
    print(model.predict(img_arr))
    return emotions[predicted_label]


if SAMPLE_TYPE == "image":
    img = cv2.imread(TEST_SAMPLE_PATH + "img/happy2.jpg", 0)
    img = cv2.equalizeHist(img) 
    img_arr = preprocess_img(img)
    predicted_emotion = get_emotion(img_arr)
    cv2.imshow(f"person is looking {predicted_emotion}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

else:
    iswebcam = True
    if SAMPLE_TYPE == "webcam":
        vidcap=cv2.VideoCapture(0)
    else:
        iswebcam = False
        vidcap=cv2.VideoCapture(TEST_SAMPLE_PATH + 'videos/video1.mp4')
        # out = cv2.VideoWriter('output.mp4', -1, 20.0, (640,480))

    while True:
        status, frame = vidcap.read()
        if not status:
            break

        if iswebcam:
            frame=cv2.flip(frame,1,0)
        
        gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = haarcascade.detectMultiScale(gray_frame, 1.32, 5)

        for face in faces:
            x, y, w, h = face
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), thickness=7)

            img = gray_frame[y:y+w, x:x+h]
            img = cv2.equalizeHist(img)
            img_arr = preprocess_img(img)

            predicted_emotion = get_emotion(img_arr)

            cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        resized_img = cv2.resize(frame, (600, 500))
        cv2.imshow('Facial Emotion Recognition',resized_img)
        # if not iswebcam:
        #     out.write(resized_img)

        if cv2.waitKey(10) == ord('q'):
            break

    vidcap.release()
    cv2.destroyAllWindows
