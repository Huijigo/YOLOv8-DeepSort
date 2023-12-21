import pandas as pd
from ultralytics import YOLO
import os
import cv2

path = r'/home/lsh/Projects/Yolov5_MOT/Yolov5-deepsort-inference/TrackEval/data/gt/mot_challenge/MOT16-train/MOT16-13/img1'
model = YOLO(r'/home/lsh/Projects/Yolov5_MOT/Yolov5-deepsort-inference/weights/best.pt')
files = os.listdir(path)
files.sort()
video_output_path = './video.avi'
fps = 24.0  # 帧率
frame_size = (1920, 1080)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = cv2.VideoWriter(video_output_path, fourcc, fps, frame_size)

# 逐个读取图像并写入视频
for image_file in files:
    image_path = os.path.join(path, image_file)
    frame = cv2.imread(image_path)

    results = model(frame)
    pred_boxes = []

    for result in results:
        for box in result.boxes:
            pred_boxes.append((*(box.xyxy[0].cpu().numpy()),"person",box.conf.cpu().numpy().item()))
    
    color = (0,255,0)
    thickness = 2
    for pred_box in pred_boxes:
        start_point = (int(pred_box[0]), int(pred_box[1]))
        end_point = (int(pred_box[2]), int(pred_box[3]))
        cv2.rectangle(frame, start_point, end_point, color, thickness)
        text = f'Confidence: {pred_box[5]:.2f}'
        org = (int(pred_box[0]), int(pred_box[1]) - 10) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        cv2.putText(frame, text, org, font, font_scale, color, font_thickness)

    if frame is not None:
        # 将图像写入视频
        video_writer.write(frame)