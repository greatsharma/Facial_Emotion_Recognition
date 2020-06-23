import warnings
warnings.filterwarnings('ignore')

import cv2
import dlib
import argparse
import numpy as np
from imutils.face_utils import rect_to_bb

import utils


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True,
                help='Input, `video` and `webcam` is supported only')
args = vars(ap.parse_args())

# Opencv DNN
modelFile = "face_detectors/dnn_tf.pb"
configFile = "face_detectors/dnn_tf.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
conf_threshold = 0.7


def dnn_detector(frame_orig):
    frame = frame_orig.copy()
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
    face_img = np.zeros_like(frame) #gray_frame
    mask = np.zeros_like(gray_frame)

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
            lm = utils.get_landmarks(gray_frame, frame, utils.bb_to_rect(face), annotate=False)
            boundary_lm = lm[:27]

            hull_points = cv2.convexHull(np.array(boundary_lm, dtype=np.int32))
            cv2.fillConvexPoly(mask, hull_points, 255)
            face_img = cv2.bitwise_and(frame, frame, mask=mask) #gray_frame, gray_frame

            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
    return frame, face_img


if args["input"] == "webcam":
    iswebcam = True
    vidcap=cv2.VideoCapture(0)
else:
    iswebcam = False
    vidcap=cv2.VideoCapture(args["input"])

while True:
    status, frame = vidcap.read()
    if not status:
        break

    if iswebcam:
        frame=cv2.flip(frame,1,0)

    frame, face_img = dnn_detector(frame)
    # face_img = cv2.merge([face_img,face_img,face_img])
    combined = np.hstack((frame, face_img))

    cv2.imshow('2D Face Extraction', combined)
    if cv2.waitKey(10) == ord('q'):
        break

vidcap.release()
cv2.destroyAllWindows