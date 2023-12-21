from AIDetector_pytorch_Change import Detector
import imutils
import cv2

def main():

    func_status = {}
    func_status['headpose'] = None
    
    name = 'demo'

    det = Detector()
    cap = cv2.VideoCapture(0)
    
    
    frame = 1
    while True:

        # try:
        _, im = cap.read()
        if im is None:
            break
        
        result = det.feedCap(im,func_status,frame)
        result = result['frame']
        result = imutils.resize(result, height=500)
        frame += 1
        cv2.imshow(name, result)
        cv2.waitKey(0)

        if cv2.getWindowProperty(name, cv2.WND_PROP_AUTOSIZE) < 1:
            # 点x退出
            break
        # except Exception as e:
        #     print(e)
        #     break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    main()