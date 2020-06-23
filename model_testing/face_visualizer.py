import warnings
warnings.filterwarnings('ignore')

import cv2
import time
import joblib
import argparse
import numpy as np
from matplotlib import pyplot
from imutils.face_utils import rect_to_bb
from tensorflow.keras.models import load_model

import utils


ap = argparse.ArgumentParser()

ap.add_argument('-i', '--input', required=True,
                help='Input, `video` and `webcam` is supported only')
ap.add_argument('-m', '--model', required=True,
                help='Model to use, like CNNModel_fer_3emo')
ap.add_argument('-d', '--detector', required=True,
                help='Face detector to use, either `haar` or `dnn`')
ap.add_argument('-he', '--histogram_equalization', required=True,
                help='1 to apply histogram_equalization otherwise 0')

args = vars(ap.parse_args())
hist_eq = utils.arg2bool(args["histogram_equalization"])

# Opencv Haar
cascade_detector = cv2.CascadeClassifier("face_detectors/haarcascade_frontalface.xml")

# Opencv DNN
modelFile = "face_detectors/dnn_tf.pb"
configFile = "face_detectors/dnn_tf.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
conf_threshold = 0.7


def haar_detector(frame):
    gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_frame = np.zeros(gray_frame.shape, dtype="uint8")

    offset = 15
    x_pos,y_pos = 10,40

    faces = cascade_detector.detectMultiScale(gray_frame, 1.32, 5)
    for idx,face in enumerate(faces):
        if hist_eq:
            gray_frame = cv2.equalizeHist(gray_frame)

        img_arr = utils.align_face(gray_frame, utils.bb_to_rect(face), desiredLeftEye)
        face_frame = cv2.resize(img_arr, (48,48), interpolation=cv2.INTER_CUBIC)
        img_arr = utils.preprocess_img(img_arr, resize=False)

        predicted_proba = model.predict(img_arr)
        predicted_label = np.argmax(predicted_proba[0])

        x,y,w,h = face
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        text = f"Person {idx+1}: {label2text[predicted_label]}"
        utils.draw_text_with_backgroud(frame, text, x + 5, y, font_scale=0.4)

        text = f"Person {idx+1} :  "
        y_pos = y_pos + 2*offset
        utils.draw_text_with_backgroud(frame, text, x_pos, y_pos, font_scale=0.3, box_coords_2=(2,-2))
        for k,v in label2text.items():
            text = f"{v}: {round(predicted_proba[0][k]*100, 3)}%"
            y_pos = y_pos + offset
            utils.draw_text_with_backgroud(frame, text, x_pos, y_pos, font_scale=0.3, box_coords_2=(2,-2))
    return frame, face_frame


def dnn_detector(frame):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    idx = 0
    offset = 15
    x_pos,y_pos = 10,40

    gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_frame = np.zeros(gray_frame.shape, dtype="uint8")

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            idx += 1
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])

            face = [x1, y1, x2-x1, y2-y1]
    
            if hist_eq:
                gray_frame = cv2.equalizeHist(gray_frame)

            img_arr = utils.align_face(gray_frame, utils.bb_to_rect(face), desiredLeftEye)
            face_frame = cv2.resize(img_arr, (48,48), interpolation=cv2.INTER_CUBIC)
            img_arr = utils.preprocess_img(img_arr, resize=False)

            predicted_proba = model.predict(img_arr)
            predicted_label = np.argmax(predicted_proba[0])

            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
            text = f"Person {idx}: {label2text[predicted_label]}"
            utils.draw_text_with_backgroud(frame, text, x1 + 5, y1, font_scale=0.4)

            text = f"Person {idx} :  "
            y_pos = y_pos + 2*offset
            utils.draw_text_with_backgroud(frame, text, x_pos, y_pos, font_scale=0.3, box_coords_2=(2,-2))
            for k,v in label2text.items():
                text = f"{v}: {round(predicted_proba[0][k]*100, 3)}%"
                y_pos = y_pos + offset
                utils.draw_text_with_backgroud(frame, text, x_pos, y_pos, font_scale=0.3, box_coords_2=(2,-2))
    return frame, face_frame


desiredLeftEye=(0.31, 0.31)
model = load_model("outputs/models/" + args["model"] + ".h5")
label2text = joblib.load("outputs/label2text/label2text_" + args["model"] + ".pkl")

if args["input"] == "webcam":
    iswebcam = True
    vidcap=cv2.VideoCapture(0)
else:
    iswebcam = False
    vidcap=cv2.VideoCapture(args["input"])

frame_count = 0
total_time = 0
while True:
    status, frame = vidcap.read()
    if not status:
        break

    frame_count += 1

    if iswebcam:
        frame=cv2.flip(frame,1,0)

    tik = time.time()
    if args["detector"] == "haar":
        frame, face_frame = haar_detector(frame)
    else:
        frame, face_frame = dnn_detector(frame)
    total_time += time.time() - tik

    fps = frame_count / total_time
    label = f"Detector: {args['detector']} ; HistEq: {args['histogram_equalization']} ; FPS: {round(fps, 2)}"
    utils.draw_text_with_backgroud(frame, label, 10, 20, font_scale=0.35)

    face_frame = cv2.resize(face_frame, (640,480))
    face_frame = cv2.merge([face_frame,face_frame,face_frame])

    label = f"Detector: {args['detector']} ; HistEq: {args['histogram_equalization']}"
    utils.draw_text_with_backgroud(face_frame, label, 10, 20, font_scale=0.5)

    combined = np.hstack((frame, face_frame))
    cv2.imshow('Facial Emotion Recognition',combined)
    if cv2.waitKey(10) == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows
