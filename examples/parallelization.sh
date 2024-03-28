python track.py --yolo-model yolov8n --save-mot --source='../footage/tampines_sample.mp4' --vid_stride 5 &                      # bboxes only
python examples/track.py --yolo-model yolov8n-pose --save-mot --show --source='../footage/tampines_sample.mp4' --vid_stride 5 & # bboxes and pose keypoints only
