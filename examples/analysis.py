## Put in code to run track.py for multiple .mp4 files 
from os import listdir
from os.path import isfile, join
import torchvision
import subprocess
import torch 
import time
import os

file_directory = '/Users/shreyaskumar/Documents/Shreyas/Work/yolo_v_8_testing/yolo_tracking/ease_footage/'
footage_files = [join(file_directory, f) for f in listdir(file_directory) if isfile(join(file_directory, f))]
footage_files.sort()
print(footage_files)
print(os.getcwd())
print(isfile('track.py'))
for footage_file in footage_files[:]:
    print(footage_file)
    start_time = time.time()
    #subprocess.run(f'python examples/track.py --yolo-model yolov8s --show --save --save-mot --source={footage_file} --vid_stride 10')
    subprocess.run(['python3', './track.py', '--yolo-model', 'yolov8s', '--show', '--save', '--save-mot','--source', footage_file, '--vid_stride', '30'])
    end_time = time.time()
    print(f'Processed {footage_file} in {end_time-start_time} seconds')


print(f'Total processing time for camera footage = {end_time - start_time}')
