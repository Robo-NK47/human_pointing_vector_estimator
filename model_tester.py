import cv2
import mediapipe as mp
from os import listdir
from os.path import isfile, join
import os
import torch
import gc
from PIL import Image
import PIL
from math import atan2, degrees
import numpy as np


def get_list_of_image_paths(directory):
    only_files = [f for f in listdir(directory) if isfile(join(directory, f))]
    _paths = []

    for _file in only_files:
        if _file[-3:] == 'jpg':
            _paths.append(os.path.join(directory, _file))

    return _paths


def adjust_frame(frame, image_shape, padding_value):
    adjusted_frame = []
    image_x = image_shape[1]
    image_y = image_shape[0]

    if (frame[0] - padding_value) < 0:
        adjusted_frame.append(0)
    else:
        adjusted_frame.append(frame[0] - padding_value)

    if (frame[1] + padding_value) > image_x:
        adjusted_frame.append(image_x)
    else:
        adjusted_frame.append(frame[1] + padding_value)

    if (frame[2] - padding_value) < 0:
        adjusted_frame.append(0)
    else:
        adjusted_frame.append(frame[2] - padding_value)

    if (frame[3] + padding_value) > image_y:
        adjusted_frame.append(image_y)
    else:
        adjusted_frame.append(frame[3] + padding_value)

    return adjusted_frame


def resize_image(_image, size_factor):
    while _image.shape[0] < size_factor and _image.shape[1] < size_factor:
        _image = cv2.resize(_image, (0, 0), fx=2, fy=2)

    return _image


def apply_brightness_contrast(input_img, brightness, contrast):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf


def get_image_contrast(_image):
    img_grey = cv2.cvtColor(_image, cv2.COLOR_BGR2GRAY)
    return img_grey.std()


def get_image_brightness(_image):
    _image = Image.fromarray(_image).convert('L')
    stat = PIL.ImageStat.Stat(_image)
    return stat.mean[0]


def get_detections(directory, _yolo_model, _distance):
    all_detections = {}
    _IMAGE_FILES = get_list_of_image_paths(directory)
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                                min_detection_confidence=0.5) as _pose:
        for i, _file in enumerate(_IMAGE_FILES):
            print(_file)
            name_to_save = _file.replace(directory, '')[1:-4] + f'({_distance} meters from camera)'
            name_to_save = os.path.join(os.getcwd(), 'temp pics', name_to_save)
            _organ_dictionary = {'nose': 0, 'left_inner_eye': 1, 'left_eye': 2, 'left_eye_outer': 3,
                                 'right_inner_eye': 4, 'right_eye': 5, 'right_eye_outer': 6, 'left_ear': 7,
                                 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10, 'left_shoulder': 11,
                                 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15,
                                 'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18, 'left_index': 19,
                                 'right_index': 20, 'left_thumb': 21, 'right_thumb': 22, 'left_hip': 23,
                                 'right_hip': 24, 'left_knee': 25, 'right_knee': 26, 'left_ankle': 27,
                                 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30, 'left_foot_index': 31,
                                 'right_foot_index': 32}
            _image = cv2.imread(_file)
            _image_height, _image_width, _ = _image.shape

            cv2.imwrite(f'{name_to_save} - 0 full.png', _image)

            yolo_results = _yolo_model(_image).pandas().xyxy[0]
            yolo_results = yolo_results[yolo_results.eq('person').any(1)].sort_values('confidence').iloc[-1]
            if len(yolo_results) > 0:
                padding_value = 50
                frame = (int(yolo_results['xmin']), int(yolo_results['xmax']),
                         int(yolo_results['ymin']), int(yolo_results['ymax']))
                frame = adjust_frame(frame, _image.shape[0:2], padding_value)
                _image = _image[frame[2]:frame[3], frame[0]:frame[1]]
                cv2.imwrite(f'{name_to_save} - 1 crop.png', _image)
                _image = resize_image(_image, 500)

                image_brightness = get_image_brightness(_image)
                image_contrast = get_image_contrast(_image)

                _image = apply_brightness_contrast(_image, 150 - image_brightness,
                                                   400 * ((image_contrast / image_brightness)**2))
                cv2.imwrite(f'{name_to_save} - 2 crop resize and brighten.png', _image)
                _results = _pose.process(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB))

                if not _results.pose_landmarks:
                    all_detections[_file] = 'Failed to identify a person'
                    continue

            else:
                all_detections[_file] = None
                continue
            draw_detections_on_image(_image, f'{name_to_save} - 3 with pose landmarks.png', _results)
            for _organ in _organ_dictionary:
                _detection = _results.pose_landmarks.landmark[_organ_dictionary[_organ]]
                _organ_dictionary[_organ] = {'x': _detection.x, 'y': _detection.y,
                                             'z': _detection.z, 'visibility': _detection.visibility}

            all_detections[_file] = _organ_dictionary

    return all_detections


def draw_detections_on_image(_image, _image_path, _results):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    annotated_image = _image.copy()
    mp_drawing.draw_landmarks(annotated_image, _results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    cv2.imwrite(_image_path, annotated_image)


def get_angle(point_1, point_2):
    angle = atan2(point_1['y'] - point_2['y'], point_1['x'] - point_2['x'])
    return 180 + degrees(angle)


img_directory = r'C:\Users\kahan\PycharmProjects\pose estimation\pics'
distances = [f for f in listdir(img_directory)]
results_by_distance = {}
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')

for distance in distances:
    path = os.path.join(img_directory, distance)
    results_by_distance[distance] = get_detections(path, yolo_model, distance)

keys = list(results_by_distance.keys())
int_keys = []
for key in keys:
    int_keys.append(int(key))

scores = {}.fromkeys(sorted(int_keys))

for key in scores:
    counter = 0
    for score in results_by_distance[str(key)]:
        if results_by_distance[str(key)][score] != 'Failed to identify a person':
            counter += 1
    scores[key] = (counter, len(results_by_distance[str(key)]))

print("\n\n")
for (distance, result) in scores.items():
    print(f'Detected skeleton in {result[0]} out of {result[1]} images from {distance} meters away.')

hands_only = {}
for key in scores:
    counter = 0
    for score in results_by_distance[str(key)]:
        detection = results_by_distance[str(key)][score]
        if detection != 'Failed to identify a person':
            hands = {'left': {'shoulder': detection['left_shoulder'], 'elbow': detection['left_elbow'],
                              'wrist': detection['left_wrist'], 'index': detection['left_index']},
                     'right': {'shoulder': detection['right_shoulder'], 'elbow': detection['right_elbow'],
                               'wrist': detection['right_wrist'], 'index': detection['right_index']}}
            hands_only[score] = hands

del (detection, counter, distance, hands, int_keys, key, path, result, score, yolo_model)
gc.collect()

angles = hands_only.copy()
for file in hands_only:
    for direction in hands_only[file]:
        angles[file][direction] = get_angle(hands_only[file][direction]['elbow'], hands_only[file][direction]['wrist'])

print('a')
