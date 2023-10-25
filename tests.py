import cv2
from ultralytics import YOLO
import supervision as sv
import torch
import os
import numpy as np
# TODO: 1. rotate, 2. flip, 3. crop
# TODO: minimum 15 zdjec - nie no zart xD

# TODO: Zrobienie struktury dataset'u dla yolo
# TODO: Generowanie pliku .yaml do trenowania modelu


def save_to_dataset(image, name, coordinates):
    cv2.imwrite(f'./LedDS/train/images/{name}.jpg', image)
    with open(f'./LedDS/train/labels/{name}.txt', 'w') as file:
        file.write(
            f'0 {str(coordinates)[1:-1].replace(",", " ")}')
    pass



# torch.backends.cudnn.enabled = False
# model = YOLO(f"../yolov8m.pt")
# model.train(data="./LedAPI/ModelDatasetled/data.yaml",
#             imgsz=640,
#             epochs=5,
#             batch=8,
#             name=f"yolo-custom",)

model = YOLO('/home/filip/PycharmProjects/LedDetection/runs/detect/yolo-custom7/weights/best.pt')
box_annontator = sv.BoxAnnotator()
image = cv2.imread('le3.png')
image = cv2.resize(image, (640, 640))
result = model.predict(image)[0]
detections = sv.Detections.from_ultralytics(result)
detections = detections[detections.confidence > 0.5]
labels = [
f"solar panel {confidence:0.2f}"
for _, _, confidence, _, _
in detections
]
annotated_image = box_annontator.annotate(image, detections, labels=labels)
cv2.imshow('img', annotated_image)
cv2.waitKey(0)
# torch.backends.cudnn.enabled = False
# model = YOLO(f"yolov8m.pt")
# model.train(
#     data='/home/filip/PycharmProjects/LedDetection/LedDS/data.yaml',
#     imgsz=640,
#     epochs=10,
#     batch=8,
#     name=f"yolo-custom",
# )
"""
while True:
    cv2.imshow('image', image)
    cv2.imshow('flip', flip)
    cv2.imshow('flip2', flip2)
    cv2.imshow('rotate', rotate)
    cv2.imshow('rotateflip', rotateflip)
    cv2.imshow('rotateflip2', rotateflip2)

    if cv2.waitKey(1) == ord('q'):
        break
"""
