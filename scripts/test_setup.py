import cv2
import mediapipe as mp
from ultralytics import YOLO
import sklearn

print("OpenCV:", cv2.__version__)
print("scikit-learn:", sklearn.__version__)
model = YOLO("yolov8n.pt")
print("YOLO loaded successfully")