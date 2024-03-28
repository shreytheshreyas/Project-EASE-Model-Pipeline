# Mikel Broström 🔥 Yolo Tracking 🧾 AGPL-3.0 license

import numpy as np
import torch
import os
import csv
from ultralytics.utils import ops


def write_mot_results(txt_path, results, frame_idx, model_type):
    model_types = ['yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose', 'yolov8x-pose']
    nr_dets = len(results.boxes)
    frame_idx = torch.full((1, 1), frame_idx + 1)
    frame_idx = frame_idx.repeat(nr_dets, 1)
    dont_care = torch.full((nr_dets, 1), -1)

    mot = torch.cat([
        frame_idx, #Frame ID
        results.boxes.id.unsqueeze(1).to('cpu'), #Box ID
        ops.xyxy2ltwh(results.boxes.xyxy).to('cpu'), # Bounding Box Information: [top-left-point-x-coordinate, top-left-point-y-coordinate, box-width, box-height]
        results.boxes.conf.unsqueeze(1).to('cpu'), # Confidence Score of the bounding box classification
        results.boxes.cls.unsqueeze(1).to('cpu'), # Classification of the object present in the bounding box
        results.keypoints.xy.reshape(nr_dets,-1).to('cpu') if model_type == 'yolov8n-pose' else dont_care,
        results.keypoints.conf.to('cpu') if model_type == 'yolov8n-pose' else dont_care
        #dont_care
    ], dim=1)
    
    #print(mot)
    
    '''
    if model_type == 'yolov8n-pose':
        print('pose-estimation')
        for idx, keypoint in enumerate(results.keypoints.xy[0]):
            updatedKeypoint = keypoint.reshape((mot.size()[0],-1))
            print(f'keypoint before resizing: {keypoint}')
            print(f'keypoint after resizing : {updatedKeypoint}')
            print(f'Size of Feature vector {mot.size()[0]}')
            print(mot)
            mot = torch.cat((mot.to('cpu'), updatedKeypoint.to('cpu')), dim=1)
        print(mot)
    '''
    # create parent folder
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    # create mot txt file
    if not os.path.exists(txt_path):
        txt_path.touch(exist_ok=True)
        with open(str(txt_path), 'w', newline='') as file:  # append binary mode
            column_headings = []
            if model_type in model_types:
                column_headings = ['frame_id', 'object_id', 'top_left_point_x_coordinate', 'top_left_point_y_coordinate', 'box_width', 'box_height', 'object_confidence',
                                   'object_class', 'nose_x_coordinate', 'nose_y_coordinate', 'left_eye_x_coordinate', 'left_eye_y_coordinate', 'right_eye_x_coordinate',
                                   'right_eye_y_coordinate', 'left_ear_x_coordinate', 'left_ear_y_coordinate', 'right_ear_x_coordinate', 'right_ear_y_coordinate',
                                   'left_shoulder_x_coordinate', 'left_shoulder_y_coordinate', 'right_shoulder_x_coordinate', 'right_shoulder_y_coordinate', 'left_elbow_x_coordinate',
                                   'left_elbow_y_coordinate', 'right_elbow_x_coordinate', 'right_elbow_y_coordinate', 'left_wrist_x_coordinate', 'left_wrist_y_coordinate', 'right_wrist_x_coordinate',
                                   'right_wrist_y_coordinate', 'left_hip_x_coordinate', 'left_hip_y_coordinate', 'right_hip_x_coordinate', 'right_hip_y_coordinate', 'left_knee_x_coordinate', 'left_knee_y_coordinate',
                                   'right_knee_x_coordinate', 'right_knee_y_coordinate', 'left_ankle_x_coordinate', 'left_ankle_y_coordinate', 'right_ankle_x_coordinate', 'right_ankle_y_coordinate',
                                   'confidence_nose', 'confidence_left_eye', 'confidence_right_eye', 'confidence_left_ear', 'confidence_right_ear', 'confidence_left_shoulder', 'confidence_right_shoulder',
                                   'confidence_left_elbow', 'confidence_right_elbow', 'confidence_left_wrist', 'confidence_right_wrist', 'confidence_left_hip', 'confidence_right_hip', 'confidence_left_knee',
                                   'confidence_right_knee', 'confidence_left_ankle', 'confidence_right_ankle']
            else:
                column_headings = ['frame_id', 'object_id', 'top_left_point_x_coordinate', 'top_left_point_y_coordinate', 'box_width', 'box_height', 'object_confidence',
                                   'object_class', 'dont_care', 'dont_care']

            file_writer = csv.writer(file)
            file_writer.writerow(column_headings)
            


    with open(str(txt_path), 'ab+') as f:  # append binary mode
        np.savetxt(f, mot.numpy(), fmt='%.2f', delimiter=',')  # save as ints instead of scientific notation
