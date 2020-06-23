import cv2
import math
import dlib
import numpy as np
from collections import OrderedDict
from imutils.face_utils import FaceAligner


face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("face_detectors/shape_predictor_68_face_landmarks.dat")

key_landmarks = {
    "kl_1": list(range(17,28)),
    "kl_2": [37,38,40,41],
    "kl_3": [43,44,46,47],
    "kl_4": [61,62,63],
    "kl_5": [65,66,67],
    "kl_6": [2,3,4,5,48],
    "kl_7": [11,12,13,14,54],
    "kl_8": [19,36,39],
    "kl_9": [24,42,45],
    "kl_10": [27,28,29,30],
    "kl_11": list(range(48, 60)),
}

ROI_1 = list(set(range(17,48)) - set(range(29,36)))
ROI_2 = list(range(48,68)) + [33, 4, 14]


def preprocess_img(img, resize):
    if resize:
        img = cv2.resize(img, (48,48), interpolation=cv2.INTER_CUBIC)
    img = img / 255.
    img = np.expand_dims(img, axis = 2)
    img = np.expand_dims(img, axis = 0)
    return img


def bb_to_rect(bb):
    left = bb[0]
    top = bb[1]
    right = bb[2]+bb[0]
    bottom = bb[3]+bb[1]
    return dlib.rectangle(left, top, right, bottom)


def align_face(gray_frame, face, desiredLeftEye):
    face_aligner = FaceAligner(shape_predictor, desiredLeftEye=desiredLeftEye, desiredFaceWidth=48)
    aligned_face = face_aligner.align(gray_frame, gray_frame, face)
    return aligned_face


def get_landmarks(gray_frame, frame, face, point_color=(0, 0, 255), point_thickness=2, annotate=False):
    landmarks = shape_predictor(gray_frame, face)
    landmarks_coord = []
    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        landmarks_coord.append((x,y))
        if annotate:
            cv2.circle(frame, (x, y), point_thickness, point_color, -1)
    return landmarks_coord


def rect_contains(rect, point):
# Check if a point is inside a rectangle
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
	 

def draw_delaunay(img, subdiv, line_color, line_thickness):
    triangleList = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangleList :
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])
        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3) :
            cv2.line(img, pt1, pt2, line_color, line_thickness)
            cv2.line(img, pt2, pt3, line_color, line_thickness)
            cv2.line(img, pt3, pt1, line_color, line_thickness)


def annotate_delaunay_triangulation(frame, landmarks_coord, line_color=(59,181,207), line_thickness=1):
    size = frame.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect)
    
    for coords in landmarks_coord:
        subdiv.insert(coords)
    draw_delaunay(frame, subdiv, line_color, line_thickness)


def get_keylandmarks_coords(landmarks_coord):
    key_landmarks_coords = {}
    for i in range(1,12):
        key_landmarks_coords[f"kl_{i}"] = []

    for i in range(0, 68):
        for k in key_landmarks:
            if i in key_landmarks[k]:
                key_landmarks_coords[k].append(landmarks_coord[i])
    
    return key_landmarks_coords


def annotate_ROI(frame, landmarks_coord, line_color=(59,181,207), line_thickness=1):
    key_landmarks_coords = get_keylandmarks_coords(landmarks_coord)

    for coord in key_landmarks_coords["kl_1"][:-1]:
        cv2.line(frame, coord, key_landmarks_coords["kl_1"][-1], line_color, line_thickness)

    for coord1, coord2 in zip(key_landmarks_coords["kl_2"][:2], key_landmarks_coords["kl_2"][::-1][:2]):
        cv2.line(frame, coord1, coord2, (0,0,255), 2)

    for coord1, coord2 in zip(key_landmarks_coords["kl_3"][:2], key_landmarks_coords["kl_3"][::-1][:2]):
        cv2.line(frame, coord1, coord2, (0,0,255), 2)

    for coord1, coord2 in zip(key_landmarks_coords["kl_4"], key_landmarks_coords["kl_5"][::-1]):
        cv2.line(frame, coord1, coord2, (0,0,255), 2)

    for coord in key_landmarks_coords["kl_6"][:-1]:
        cv2.line(frame, coord, key_landmarks_coords["kl_6"][-1], line_color, line_thickness)

    for coord in key_landmarks_coords["kl_7"][:-1]:
        cv2.line(frame, coord, key_landmarks_coords["kl_7"][-1], line_color, line_thickness)

    for coord in key_landmarks_coords["kl_8"][1:]:
        cv2.line(frame, coord, key_landmarks_coords["kl_8"][0], (0,255,0), 1)
    
    for coord in key_landmarks_coords["kl_9"][1:]:
        cv2.line(frame, coord, key_landmarks_coords["kl_9"][0], (0,255,0), 1)

    for i,coord1 in enumerate(key_landmarks_coords["kl_10"]):
        for coord2 in key_landmarks_coords["kl_11"]:
            cv2.line(frame, coord1, coord2, line_color, line_thickness)


def arg2bool(x):
    if x == '1':
        return True
    elif x == '0':
        return False
    else:
        raise ValueError(f'bool argument should be either 1 or 0 but got {x}')


def draw_text_with_backgroud(img, text, x, y, font_scale, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX,
                            background=(0,0,0), foreground=(255,255,255), box_coords_1=(-5,5), box_coords_2=(5,-5)):
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
    box_coords = ((x+box_coords_1[0], y+box_coords_1[1]), (x + text_width + box_coords_2[0], y - text_height + box_coords_2[1]))
    cv2.rectangle(img, box_coords[0], box_coords[1], background, cv2.FILLED)
    cv2.putText(img, text, (x, y), font, fontScale=font_scale, color=foreground, thickness=thickness)


def euclidean_dist(point_1: tuple, point_2: tuple):
    return math.sqrt((point_1[0]-point_2[0])**2 + (point_1[1]-point_2[1])**2)


def extract_hog_features(gray_img):
    img_ = cv2.resize(gray_img, (64,128))
    hog = cv2.HOGDescriptor()
    hog_feature = hog.compute(img_)
    return hog_feature


def extract_roi1_roi2(gray_img, landmarks):
    ROI1_landmarks = []
    ROI2_landmarks = []

    for i in range(0, 68):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        if i in ROI_1:
            ROI1_landmarks.append((x,y))
        if i in ROI_2:
            ROI2_landmarks.append((x,y))

    (x, y, w, h) = cv2.boundingRect(np.array(ROI1_landmarks))
    roi1 = gray_img[y:y+h, x:x+w]
    roi1 = cv2.resize(roi1, (50,25), interpolation=cv2.INTER_CUBIC)

    (x, y, w, h) = cv2.boundingRect(np.array(ROI2_landmarks))
    roi2 = gray_img[y:y+h, x:x+w]
    roi2 = cv2.resize(roi2, (50,25), interpolation=cv2.INTER_CUBIC)

    return np.expand_dims(roi1, axis=2), np.expand_dims(roi2, axis=2)


def get_keylandmarks_distances(key_landmarks_coords):
    key_landmarks_distance = {}

    key_landmarks_distance["kl_dist_1"] = []
    for coords in key_landmarks_coords["kl_1"][:-1]:
        key_landmarks_distance["kl_dist_1"].append(euclidean_dist(coords, key_landmarks_coords["kl_1"][-1]))

    key_landmarks_distance["kl_dist_2"] = []
    for coord1, coord2 in zip(key_landmarks_coords["kl_2"][:2], key_landmarks_coords["kl_2"][::-1][:2]):
        key_landmarks_distance["kl_dist_2"].append(euclidean_dist(coord1, coord2))

    key_landmarks_distance["kl_dist_3"] = []
    for coord1, coord2 in zip(key_landmarks_coords["kl_3"][:2], key_landmarks_coords["kl_3"][::-1][:2]):
        key_landmarks_distance["kl_dist_3"].append(euclidean_dist(coord1, coord2))

    key_landmarks_distance["kl_dist_4"] = []
    for coord1, coord2 in zip(key_landmarks_coords["kl_4"], key_landmarks_coords["kl_5"][::-1]):
        key_landmarks_distance["kl_dist_4"].append(euclidean_dist(coord1, coord2))

    key_landmarks_distance["kl_dist_5"] = []
    for coord in key_landmarks_coords["kl_6"][:-1]:
        key_landmarks_distance["kl_dist_5"].append(euclidean_dist(coord, key_landmarks_coords["kl_6"][-1]))

    key_landmarks_distance["kl_dist_6"] = []
    for coord in key_landmarks_coords["kl_7"][:-1]:
        key_landmarks_distance["kl_dist_6"].append(euclidean_dist(coord, key_landmarks_coords["kl_7"][-1]))

    key_landmarks_distance["kl_dist_7"] = []
    for coord in key_landmarks_coords["kl_8"][1:]:
        key_landmarks_distance["kl_dist_7"].append(euclidean_dist(coord, key_landmarks_coords["kl_8"][0]))

    key_landmarks_distance["kl_dist_8"] = []
    for coord in key_landmarks_coords["kl_9"][1:]:
        key_landmarks_distance["kl_dist_8"].append(euclidean_dist(coord, key_landmarks_coords["kl_9"][0]))

    for i,coord1 in enumerate(key_landmarks_coords["kl_10"]):
        key_landmarks_distance[f"kl_dist_{8+i+1}"] = []
        for coord2 in key_landmarks_coords["kl_11"]:
            key_landmarks_distance[f"kl_dist_{8+i+1}"].append(euclidean_dist(coord1,coord2))
    
    key_landmarks_distance = OrderedDict(sorted(key_landmarks_distance.items(), key=lambda i: int(i[0].split("_")[-1])))
    all_kl_dists = [i for v in key_landmarks_distance.values() for i in v]
    return all_kl_dists