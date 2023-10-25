import cv2
from ultralytics import YOLO
import torch
import os
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


def create_coords(coordinates):
    data = {
            "basic": coordinates,
            "flip": [640 - coordinates[0], coordinates[1], 640 - coordinates[2], coordinates[3]],
            "flip2": [coordinates[0], 640 - coordinates[1], coordinates[2], 640 - coordinates[3]],
            "rotate": [640 - coordinates[0], 640 - coordinates[1], 640 - coordinates[2], 640 - coordinates[3]],
            "rotate2": [640 - coordinates[1], 640 - coordinates[0], 640 - coordinates[3], 640 - coordinates[2]],
            "rotate3": [coordinates[1], coordinates[0], coordinates[3], coordinates[2]],
            "basic_blur": coordinates,
            "flip_blur": [640 - coordinates[0], coordinates[1], 640 - coordinates[2], coordinates[3]],
            "flip2_blur": [coordinates[0], 640 - coordinates[1], coordinates[2], 640 - coordinates[3]],
            "rotate_blur": [640 - coordinates[0], 640 - coordinates[1], 640 - coordinates[2], 640 - coordinates[3]],
            "rotate2_blur": [640 - coordinates[1], 640 - coordinates[0], 640 - coordinates[3], 640 - coordinates[2]],
            "rotate3_blur": [coordinates[1], coordinates[0], coordinates[3], coordinates[2]]
    }
    return data



# image = cv2.imread('led.jpg')
# image = cv2.resize(image, (640, 640))
# basic_coords = create_coords([15, 75, 120, 200])
# print(basic_coords)
# normalize_coords = normalize_coordinates(basic_coords)
# print(normalize_coords)
# flip = cv2.flip(image,  1)
# flip2 = cv2.flip(image, 0)
# rotate = cv2.rotate(image, 1)
# rotateflip = cv2.rotate(flip, cv2.ROTATE_90_CLOCKWISE)
# rotateflip2 = cv2.rotate(flip, cv2.ROTATE_90_COUNTERCLOCKWISE)
# blurred = cv2.GaussianBlur(image, (5, 5), 0)
# blurred_2 = cv2.GaussianBlur(flip, (5, 5), 0)
# blurred_3 = cv2.GaussianBlur(flip2, (5, 5), 0)
# blurred_4 = cv2.GaussianBlur(rotate, (5, 5), 0)
# blurred_5 = cv2.GaussianBlur(rotateflip, (5, 5), 0)
# blurred_6 = cv2.GaussianBlur(rotateflip2, (5, 5), 0)
# save_to_dataset(image, 'led', normalize_coords['basic'])
# save_to_dataset(flip, 'flip', normalize_coords['flip'])
# save_to_dataset(flip2, 'flip2', normalize_coords['flip2'])
# save_to_dataset(rotate, 'rotate', normalize_coords['rotate'])
# save_to_dataset(rotate, 'rotate2', normalize_coords['rotate2'])
# save_to_dataset(rotate, 'rotate3', normalize_coords['rotate3'])
# save_to_dataset(blurred, 'led_blur', normalize_coords['basic_blur'])
# save_to_dataset(blurred_2, 'flip_blur', normalize_coords['flip_blur'])
# save_to_dataset(blurred_3, 'flip2_blur', normalize_coords['flip2_blur'])
# save_to_dataset(blurred_4, 'rotate_blur', normalize_coords['rotate_blur'])
# save_to_dataset(blurred_5, 'rotate2_blur', normalize_coords['rotate2_blur'])
# save_to_dataset(blurred_6, 'rotate3_blur', normalize_coords['rotate3_blur'])
#
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
