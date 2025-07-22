
# Fire Detection Using YOLOv8 and OpenCV

This project demonstrates how to train and deploy a real-time fire detection system using [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) and OpenCV. It uses a custom dataset and processes video frames to identify fire with bounding boxes and confidence scores.

---

## Installation

Install the required Python libraries:

```bash
!pip install ultralytics
!pip install cvzone

Model Training
We use YOLOv8 for object detection. The nano version yolov8n.pt is selected for fast inference and minimal resource usage.
!yolo task=detect mode=train model=yolov8n.pt data=/content/drive/MyDrive/fire/data.yaml epochs=10 imgsz=640
Arguments:
•	task=detect: Object detection task.
•	model=yolov8n.pt: Pre-trained nano model.
•	data: Path to the custom dataset YAML file.
•	epochs=10: Number of training epochs.
•	imgsz=640: Image resolution for training.

Real-Time Fire Detection
After training, you can detect fire in video footage using the trained model.
from ultralytics import YOLO
from google.colab.patches import cv2_imshow
import cv2
import cvzone
import math
Load model and video:
cap = cv2.VideoCapture('/content/fire2.mp4')
model = YOLO('/content/runs/detect/train/weights/best.pt')
classnames = ['fire']
Process frames:
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (648, 488))
    result = model(frame, stream=True)

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)

    cv2_imshow(frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

Features
•	Real-time video analysis.
•	Custom dataset support.
•	Lightweight model (YOLOv8-nano) for speed.
•	Clean bounding boxes with confidence display using cvzone.

Folder Structure
project/
│
├── data.yaml                     # Dataset configuration
├── /runs/detect/train/          # YOLO training outputs
│   └── weights/best.pt          # Best trained model
├── fire2.mp4                    # Input video file
└── fire_detection.py            # Python script

Resources
•	Ultralytics YOLOv8 Documentation
•	cvzone GitHub
•	OpenCV Python

Notes
•	This code is optimized for use in Google Colab.
•	Update file paths if running locally or on another environment.
•	You can expand this model for smoke, human, or hazard detection with additional classes in your dataset.

Author
Developed by Lama Alzu’bi
Bachelor of Artificial Intelligence Sciences
Specialized in Deep Learning & Computer Vision


