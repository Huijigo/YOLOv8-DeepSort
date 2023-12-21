import torch
import numpy as np
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, letterbox
from utils.torch_utils import select_device
from utils.BaseDetector import baseDet
from ultralytics import YOLO

class Detector(baseDet):

    def __init__(self):
        super(Detector, self).__init__()
        self.init_model()
        self.build_config()


    #加载数据集
    def init_model(self):
        self.model = YOLO(r'./weights/best.pt')
      
    def preprocess(self, img):

        img0 = img.copy()
        img = letterbox(img, new_shape=self.img_size)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half()  # 半精度
        img /= 255.0  # 图像归一化
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        return img0, img

    def detect(self, im):

        #im0, img = self.preprocess(im)

        #pred = self.m(img, augment=False)[0]
        results = self.model(im,conf=0.5)
        # pred = pred.float()
        # pred = non_max_suppression(pred, self.threshold, 0.4)


        pred_boxes = []

        for result in results:
            for box in result.boxes:
                pred_boxes.append((*(box.xyxy[0].cpu().numpy()),"person",box.conf.cpu().numpy().item()))
    
        return im, pred_boxes

