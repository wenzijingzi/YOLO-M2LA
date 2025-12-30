import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('E:/track/YOLOv11/runs/train/exp65/weights/best.pt') # select your model.pt path
    model = YOLO('yolo11x.pt')
    model.track(source='cut_out1.mp4',
                imgsz=640,
                project='runs/track',
                name='exp',
                save=True
                )