import cv2
import dlib
import time
import argparse
import numpy as np
from imutils.face_utils import rect_to_bb

import utils


ap = argparse.ArgumentParser()

ap.add_argument('-i', '--input', required=True,
                help='Input, `video` and `webcam` is supported only')
ap.add_argument('-bb', '--bounding_box', required=False,
                help='1 to show bounding box otherwise 0, default is 1')
ap.add_argument('-lm', '--landmarks', required=False,
                help='1 to show landmarks otherwise 0, default is 0')
ap.add_argument('-dt', '--delaunay_triangulation', required=False,
                help='1 to show delaunay triangulation otherwise 0, default is 0')
ap.add_argument('-roi', '--region_of_interest', required=False,
                help='1 to show region_of_interest otherwise 0, default is 0')

args = vars(ap.parse_args())

DEFAULT_PARAMS = {
    'bounding_box': (True, bool),
    'landmarks': (False, bool),
    'delaunay_triangulation': (False, bool),
    'region_of_interest': (False, bool),
}

for k in args:
    if not k in ["input", "skip_rate"]:
        args[k] = (
            DEFAULT_PARAMS[k][0]
            if args[k] is None else
            utils.arg2bool(args[k])
        )
    if k == "skip_rate":
        args["skip_rate"] = (
            1
            if args["skip_rate"] is None else
            int(args["skip_rate"])
        )

# Opencv Haar
haar_detector = cv2.CascadeClassifier("face_detectors/haarcascade_frontalface.xml")

# Opencv LBP
lbp_detector = cv2.CascadeClassifier("face_detectors/lbpcascade_frontalface.xml")

# DLIB HoG
hog_detector = dlib.get_frontal_face_detector()

# Opencv DNN
modelFile = "face_detectors/dnn_tf.pb"
configFile = "face_detectors/dnn_tf.pbtxt"
net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)
conf_threshold = 0.7


def cascade_detector(frame_orig, detector):
    if not detector in ["haar", "lbp"]:
        raise ValueError("Invalid cascade detector")
    
    face_detector = (
        haar_detector
        if detector == "haar" else
        lbp_detector
    )

    frame = frame_orig.copy()
    gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detector.detectMultiScale(gray_frame, 1.32, 5)
    for face in faces:
        landmarks_coord = utils.get_landmarks(gray_frame, frame, utils.bb_to_rect(face), annotate=args["landmarks"])
        if args["delaunay_triangulation"]:
            utils.annotate_delaunay_triangulation(frame, landmarks_coord, line_thickness=2)
        if args["region_of_interest"]:
            utils.annotate_ROI(frame, landmarks_coord)
        if args["bounding_box"]:
            x,y,w,h = face
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    return frame


def dlib_detector(frame_orig):
    frame = frame_orig.copy()
    gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = hog_detector(gray_frame)
    for face in faces:
        landmarks_coord = utils.get_landmarks(gray_frame, frame, face, annotate=args["landmarks"])
        if args["delaunay_triangulation"]:
            utils.annotate_delaunay_triangulation(frame, landmarks_coord, line_thickness=2)
        if args["region_of_interest"]:
            utils.annotate_ROI(frame, landmarks_coord)
        if args["bounding_box"]:
            x,y,w,h = rect_to_bb(face)
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
    return frame


def dnn_detector(frame_orig):
    frame = frame_orig.copy()
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            bboxes.append([x1, y1, x2, y2])

            face = [x1, y1, x2-x1, y2-y1]
    
            gray_frame= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            landmarks_coord = utils.get_landmarks(gray_frame, frame, utils.bb_to_rect(face), annotate=args["landmarks"])
            if args["delaunay_triangulation"]:
                utils.annotate_delaunay_triangulation(frame, landmarks_coord, line_thickness=2)
            if args["region_of_interest"]:
                utils.annotate_ROI(frame, landmarks_coord)
            if args["bounding_box"]:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
    return frame


if __name__ == "__main__":
    if args["input"] == "webcam":
        iswebcam = True
        vidcap=cv2.VideoCapture(0)
    else:
        iswebcam = False
        vidcap=cv2.VideoCapture(args["input"])

    cv2.namedWindow("Face Detection Comparison", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Face Detection Comparison", 1400, 700)

    frame_count = 0
    tt_haar = 0
    tt_lbp = 0
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
            out_haar = cascade_detector(frame, "haar")
            tt_haar += time.time() - tik
            fps_haar = frame_count / tt_haar
            label = f"Detector: haar ; FPS: {round(fps_haar, 2)}"
            cv2.putText(out_haar, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            
            tik = time.time()
            out_lbp = cascade_detector(frame, "lbp")
            tt_lbp += time.time() - tik
            fps_lbp = frame_count / tt_lbp
            label = f"Detector: lbp ; FPS: {round(fps_lbp, 2)}"
            cv2.putText(out_lbp, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            tik = time.time()
            out_dlib = dlib_detector(frame)
            tt_dlib += time.time() - tik
            fps_dlib = frame_count / tt_dlib
            label = f"Detector: dlib ; FPS: {round(fps_dlib, 2)}"
            cv2.putText(out_dlib, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            tik = time.time()
            out_dnn = dnn_detector(frame)
            tt_dnn += time.time() - tik
            fps_dnn = frame_count / tt_dnn
            label = "Detector: DNN_TF ; FPS : {:.2f}".format(fps_dnn)
            cv2.putText(out_dnn, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            top = np.hstack([out_haar, out_lbp])
            bottom = np.hstack([out_dlib, out_dnn])
            frame = np.vstack([top, bottom])
            
        except Exception as e:
            print(e)
            pass

        cv2.imshow("Face Detection Comparison", frame)
        if cv2.waitKey(10) == ord('q'):
            break

    cv2.destroyAllWindows()
    vidcap.release()
