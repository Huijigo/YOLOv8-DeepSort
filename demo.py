from AIDetector_pytorch_Change import Detector
import imutils
import cv2
import os


def main():

    func_status = {}
    func_status['headpose'] = None
    
    det = Detector()
    wenjians = ["MOT16-"+ i for i in ("02","04","05","09","10","11","13")]
    for wenjian in wenjians:
        path = r'/home/lsh/Projects/Yolov5_MOT/Yolov5-deepsort-inference/TrackEval/data/gt/mot_challenge/MOT16-train/'+wenjian+'/img1'
        list_path = os.listdir(path)
        list_path = sorted(list_path)
        frame = 1
        for file in list_path:
            location = os.path.join(path,file)
            print(location)
        
            im = cv2.imread(location)
            if im is None:
                break     
            result = det.feedCap(im, func_status,frame,wenjian,)
            frame += 1

if __name__ == '__main__':
    
    main()