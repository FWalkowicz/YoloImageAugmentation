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

# model = YOLO('/home/filip/PycharmProjects/LedDetection/runs/detect/yolo-custom7/weights/best.pt')
# box_annontator = sv.BoxAnnotator()
# image = cv2.imread('le3.png')
# image = cv2.resize(image, (640, 640))
# result = model.predict(image)[0]
# detections = sv.Detections.from_ultralytics(result)
# detections = detections[detections.confidence > 0.5]
# labels = [
# f"solar panel {confidence:0.2f}"
# for _, _, confidence, _, _
# in detections
# ]
# annotated_image = box_annontator.annotate(image, detections, labels=labels)
# cv2.imshow('img', annotated_image)
# cv2.waitKey(0)
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
coordinates = [[110, 160, 235, 380], [245, 280, 220, 65], [100, 50, 200, 50]]
new_coordinates = {
            "basic": coordinates,
            "flip": [],
            "flip2": [],
            "rotate": [],
            "rotate2": [],
            "rotate3": [],
            "basic_blur": coordinates,
            "flip_blur": [],
            "flip2_blur": [],
            "rotate_blur": [],
            "rotate2_blur": [],
            "rotate3_blur": [],
            "noise_1": [],
            "noise_2": [],
            "noise_3": [],
            "noise_4": [],
            "noise_5": [],
            "noise_6": [],
            "noise_7": [],
            "noise_8": [],
            "noise_9": [],
            "noise_10": [],
            "noise_11": [],
        }

for coord in coordinates:
    new_coordinates['flip'].append([
                640 - coord[0],
                coord[1],
                640 - coord[2],
                coord[3],
            ])
    new_coordinates['flip2'].append(
        [
            coord[0],
            640 - coord[1],
            coord[2],
            640 - coord[3],
        ],
    )
    new_coordinates['rotate'].append(
        [
            640 - coord[0],
            640 - coord[1],
            640 - coord[2],
            640 - coord[3],
        ]
    )
    new_coordinates['rotate2'].append(
        [
            640 - coord[1],
            640 - coord[0],
            640 - coord[3],
            640 - coord[2],
        ]
    )
    new_coordinates['rotate3'].append(
        [
            coord[1],
            coord[0],
            coord[3],
            coord[2]
        ]
    )
    new_coordinates['flip_blur'].append(
        [
            640 - coord[0],
            coord[1],
            640 - coord[2],
            coord[3],
        ]
    )
    new_coordinates['flip2_blur'].append(
        [
            coord[0],
            640 - coord[1],
            coord[2],
            640 - coord[3],
        ]
    )
    new_coordinates['rotate_blur'].append(
        [
            640 - coord[0],
            640 - coord[1],
            640 - coord[2],
            640 - coord[3],
        ]
    )
    new_coordinates['rotate2_blur'].append(
        [
            640 - coord[1],
            640 - coord[0],
            640 - coord[3],
            640 - coord[2],
        ]
    )
    new_coordinates['rotate3_blur'].append(
        [
            coord[1],
            coord[0],
            coord[3],
            coord[2],
        ]
    )
new_coordinates['noise_1'] = new_coordinates['flip']
new_coordinates['noise_2'] = new_coordinates['flip2']
new_coordinates['noise_3'] = new_coordinates['rotate']
new_coordinates['noise_4'] = new_coordinates['rotate2']
new_coordinates['noise_5'] = new_coordinates['rotate3']
new_coordinates['noise_6'] = new_coordinates['basic']
new_coordinates['noise_7'] = new_coordinates['flip_blur']
new_coordinates['noise_8'] = new_coordinates['flip2_blur']
new_coordinates['noise_9'] = new_coordinates['rotate_blur']
new_coordinates['noise_10'] = new_coordinates['rotate2_blur']
new_coordinates['noise_11'] = new_coordinates['rotate3_blur']

for coordinate in new_coordinates:
    normalized_coords = []
    for cords in new_coordinates[coordinate]:
        x1, y1 = cords[0], cords[1]
        x2, y2 = cords[2], cords[3]
        image_width, image_height = 640, 640
        normalize_x = abs((x1 + x2) / (2 * image_width))
        normalize_y = abs((y1 + y2) / (2 * image_height))
        normalize_width = abs((x2 - x1) / image_width)
        normalize_height = abs((y2 - y1) / image_height)
        normalized_coords.append([
            normalize_x,
            normalize_y,
            normalize_width,
            normalize_height,
        ])
    new_coordinates[coordinate] = normalized_coords

print(new_coordinates)

