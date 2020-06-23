import cv2
import dlib
import time
import joblib
import argparse
import numpy as np
from imutils.face_utils import rect_to_bb
from tensorflow.keras.models import load_model

import utils


ap = argparse.ArgumentParser()

ap.add_argument('-i', '--input', required=True,
                help='Input, `video` and `webcam` is supported only')
ap.add_argument('-m', '--model', required=True,
                help='Model to use, like CNNModel_fer_3emo')
ap.add_argument('-he', '--histogram_equalization', required=True,
                help='1 to apply histogram_equalization otherwise 0')

args = vars(ap.parse_args())
hist_eq = utils.arg2bool(args["histogram_equalization"])

# DLIB HoG
hog_detector = dlib.get_frontal_face_detector()

# Opencv DNN
modelFile = "face_detectors/dnn_tf.pb"
configFile = "face_detectors/dnn_tf.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
conf_threshold = 0.7


def dlib_detector(frame_orig):
    frame = frame_orig.copy()
    gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    offset = 15
    x_pos,y_pos = 10,40

    faces = hog_detector(gray_frame)
    for idx, face in enumerate(faces):
        if hist_eq:
            gray_frame = cv2.equalizeHist(gray_frame)

        img_arr = utils.align_face(gray_frame, face, desiredLeftEye)
        img_arr = utils.preprocess_img(img_arr, resize=False)

        predicted_proba = model.predict(img_arr)
        predicted_label = np.argmax(predicted_proba[0])

        x,y,w,h = rect_to_bb(face)
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

    return frame


def dnn_detector(frame_orig):
    frame = frame_orig.copy()
    gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    idx = 0
    offset = 15
    x_pos,y_pos = 10,40

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

    return frame


desiredLeftEye=(0.31, 0.31)
model = load_model("outputs/models/" + args["model"] + ".h5")
label2text = joblib.load("outputs/label2text/label2text_" + args["model"] + ".pkl")


if __name__ == "__main__":
    if args["input"] == "webcam":
        iswebcam = True
        vidcap=cv2.VideoCapture(0)
    else:
        iswebcam = False
        vidcap=cv2.VideoCapture(args["input"])

    _, frame = vidcap.read()
    cv2.namedWindow("Face Detection Comparison", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Detection Comparison", 1400, 700)

    frame_count = 0
    tt_dlib = 0
    tt_dnn = 0

    while True:
        status, frame = vidcap.read()
        if not status:
            break

        frame_count += 1

        if iswebcam:
            frame=cv2.flip(frame,1,0)

        try:
            tik = time.time()
            out_dlib = dlib_detector(frame)
            tt_dlib += time.time() - tik
            fps_dlib = frame_count / tt_dlib
            label = f"Detector: dlib ; HistEq: {args['histogram_equalization']} ; FPS: {round(fps_dlib, 2)}"
            utils.draw_text_with_backgroud(out_dlib, label, 10, 20, font_scale=0.35)

            tik = time.time()
            out_dnn = dnn_detector(frame)
            tt_dnn += time.time() - tik
            fps_dnn = frame_count / tt_dnn
            label = f"Detector: dnn_tf ; HistEq: {args['histogram_equalization']} ; FPS: {round(fps_dnn, 2)}"
            utils.draw_text_with_backgroud(out_dnn, label, 10, 20, font_scale=0.35)

            frame = np.hstack([out_dlib, out_dnn])
        except  Exception as e:
            print(e)
            pass

        cv2.imshow("Face Detection Comparison", frame)
        if cv2.waitKey(10) == ord('q'):
            break

    cv2.destroyAllWindows()
    vidcap.release()
