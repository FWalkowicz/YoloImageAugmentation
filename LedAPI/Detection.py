from abc import ABC, abstractmethod
import os
import cv2
from ultralytics import YOLO
import torch
import numpy as np


class Detection(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def create_dataset(self):
        pass

    @abstractmethod
    def normalize_coordinates(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class LedDetection(Detection):
    def __init__(self, first_image, coordinates):
        self.first_image = cv2.resize(first_image, (640, 640))
        self.image_set = [self.first_image]
        self.basic_coordinates = coordinates
        self.coordinates = None
        self.name = "led"

    def train_model(self):
        torch.backends.cudnn.enabled = False
        model = YOLO(f"yolov8m.pt")
        model.train(data="./LedAPI/ModelDatasetled/data.yaml",
                    imgsz=640,
                    epochs=5,
                    batch=8,
                    name=f"yolo-custom", )

    def create_dataset(self):
        self.create_dataset_folder()
        self.create_coords()
        self.normalize_coordinates()
        self.create_images()

    def predict(self):
        pass

    def normalize_coordinates(self) -> None:
        for coordinate in self.coordinates:
            normalized_coords = []
            for cords in self.coordinates[coordinate]:
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
            self.coordinates[coordinate] = normalized_coords
        print(self.coordinates)

    def create_dataset_folder(self):
        path = f"./ModelDataset{self.name}"
        folders = ["train", "test", "valid"]
        image_and_label_folders = ["images", "labels"]
        configuration_file = "data.yaml"
        yaml_data = """train: ./train/images
val: ./valid/images
test: ./test/images

nc: 1
names: [led]
"""
        if not os.path.exists(path):
            os.mkdir(path)
            for folder in folders:
                folder_path = os.path.join(path, folder)
                os.mkdir(folder_path)
                for subfolder in image_and_label_folders:
                    os.mkdir(os.path.join(folder_path, subfolder))
            with open(os.path.join(path, configuration_file), "w") as f:
                f.write(yaml_data)

    def create_coords(self):
        self.coordinates = {
            "basic": self.basic_coordinates,
            "flip": [],
            "flip2": [],
            "rotate": [],
            "rotate2": [],
            "rotate3": [],
            "basic_blur": self.basic_coordinates,
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
        for coord in self.basic_coordinates:
            self.coordinates['flip'].append([
                640 - coord[0],
                coord[1],
                640 - coord[2],
                coord[3],
            ])
            self.coordinates['flip2'].append(
                [
                    coord[0],
                    640 - coord[1],
                    coord[2],
                    640 - coord[3],
                ],
            )
            self.coordinates['rotate'].append(
                [
                    640 - coord[0],
                    640 - coord[1],
                    640 - coord[2],
                    640 - coord[3],
                ]
            )
            self.coordinates['rotate2'].append(
                [
                    640 - coord[1],
                    640 - coord[0],
                    640 - coord[3],
                    640 - coord[2],
                ]
            )
            self.coordinates['rotate3'].append(
                [
                    coord[1],
                    coord[0],
                    coord[3],
                    coord[2]
                ]
            )
            self.coordinates['flip_blur'].append(
                [
                    640 - coord[0],
                    coord[1],
                    640 - coord[2],
                    coord[3],
                ]
            )
            self.coordinates['flip2_blur'].append(
                [
                    coord[0],
                    640 - coord[1],
                    coord[2],
                    640 - coord[3],
                ]
            )
            self.coordinates['rotate_blur'].append(
                [
                    640 - coord[0],
                    640 - coord[1],
                    640 - coord[2],
                    640 - coord[3],
                ]
            )
            self.coordinates['rotate2_blur'].append(
                [
                    640 - coord[1],
                    640 - coord[0],
                    640 - coord[3],
                    640 - coord[2],
                ]
            )
            self.coordinates['rotate3_blur'].append(
                [
                    coord[1],
                    coord[0],
                    coord[3],
                    coord[2],
                ]
            )
        self.coordinates['noise_1'] = self.coordinates['flip']
        self.coordinates['noise_2'] = self.coordinates['flip2']
        self.coordinates['noise_3'] = self.coordinates['rotate']
        self.coordinates['noise_4'] = self.coordinates['rotate2']
        self.coordinates['noise_5'] = self.coordinates['rotate3']
        self.coordinates['noise_6'] = self.coordinates['basic']
        self.coordinates['noise_7'] = self.coordinates['flip_blur']
        self.coordinates['noise_8'] = self.coordinates['flip2_blur']
        self.coordinates['noise_9'] = self.coordinates['rotate_blur']
        self.coordinates['noise_10'] = self.coordinates['rotate2_blur']
        self.coordinates['noise_11'] = self.coordinates['rotate3_blur']

    @staticmethod
    def noise(img):
        gauss = np.random.normal(0, 1, img.size)
        gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
        return gauss

    def create_images(self):
        basic = self.first_image
        self.save_to_dataset(basic, 'basic', self.coordinates['basic'])
        flip = cv2.flip(self.first_image, 1)
        self.save_to_dataset(flip, 'flip', self.coordinates['flip'])
        flip2 = cv2.flip(self.first_image, 0)
        self.save_to_dataset(flip2, 'flip2', self.coordinates['flip2'])
        rotate = cv2.rotate(self.first_image, 1)
        self.save_to_dataset(rotate, 'rotate', self.coordinates['rotate'])
        rotateflip = cv2.rotate(flip, cv2.ROTATE_90_CLOCKWISE)
        self.save_to_dataset(rotateflip, 'rotate2', self.coordinates['rotate2'])
        rotateflip2 = cv2.rotate(flip, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.save_to_dataset(rotateflip2, 'rotate3', self.coordinates['rotate3'])
        blurred = cv2.GaussianBlur(self.first_image, (5, 5), 0)
        self.save_to_dataset(blurred, 'basic_blur', self.coordinates['basic_blur'])
        blurred_2 = cv2.GaussianBlur(flip, (5, 5), 0)
        self.save_to_dataset(blurred_2, 'flip_blur', self.coordinates['flip_blur'])
        blurred_3 = cv2.GaussianBlur(flip2, (5, 5), 0)
        self.save_to_dataset(blurred_3, 'flip2_blur', self.coordinates['flip2_blur'])
        blurred_4 = cv2.GaussianBlur(rotate, (5, 5), 0)
        self.save_to_dataset(blurred_4, 'rotate_blur', self.coordinates['rotate_blur'])
        blurred_5 = cv2.GaussianBlur(rotateflip, (5, 5), 0)
        self.save_to_dataset(blurred_5, 'rotate2_blur', self.coordinates['rotate2_blur'])
        blurred_6 = cv2.GaussianBlur(rotateflip2, (5, 5), 0)
        self.save_to_dataset(blurred_6, 'rotate3_blur', self.coordinates['rotate3_blur'])
        gauss_noise = self.noise(self.first_image)
        noise_1 = cv2.add(flip, gauss_noise)
        self.save_to_dataset(noise_1, 'noise_1', self.coordinates['noise_1'])
        noise_2 = cv2.add(flip2, gauss_noise)
        self.save_to_dataset(noise_2, 'noise_2', self.coordinates['noise_2'])
        noise_3 = cv2.add(rotate, gauss_noise)
        self.save_to_dataset(noise_3, 'noise_3', self.coordinates['noise_3'])
        noise_4 = cv2.add(rotateflip, gauss_noise)
        self.save_to_dataset(noise_4, 'noise_4', self.coordinates['noise_4'])
        noise_5 = cv2.add(rotateflip2, gauss_noise)
        self.save_to_dataset(noise_5, 'noise_5', self.coordinates['noise_5'])
        noise_6 = cv2.add(blurred, gauss_noise)
        self.save_to_dataset(noise_6, 'noise_6', self.coordinates['noise_6'])
        noise_7 = cv2.add(blurred_2, gauss_noise)
        self.save_to_dataset(noise_7, 'noise_7', self.coordinates['noise_7'])
        noise_8 = cv2.add(blurred_3, gauss_noise)
        self.save_to_dataset(noise_8, 'noise_8', self.coordinates['noise_8'])
        noise_9 = cv2.add(blurred_4, gauss_noise)
        self.save_to_dataset(noise_9, 'noise_9', self.coordinates['noise_9'])
        noise_10 = cv2.add(blurred_5, gauss_noise)
        self.save_to_dataset(noise_10, 'noise_10', self.coordinates['noise_10'])
        noise_11 = cv2.add(blurred_6, gauss_noise)
        self.save_to_dataset(noise_11, 'noise_11', self.coordinates['noise_11'])
        self.save_to_valid(noise_11, 'noise_11', self.coordinates['noise_11'])

    def save_to_dataset(self, image, name, coordinates):
        cv2.imwrite(f'./ModelDatasetled/train/images/{name}.jpg', image)
        for cords in self.coordinates[name]:
            with open(f'./ModelDatasetled/train/labels/{name}.txt', 'a') as file:
                file.write(
                    f'0 {str(cords)[1:-1].replace(",", " ")} \n')
            pass

    def save_to_valid(self, image, name, coordinates):
        cv2.imwrite(f'./ModelDatasetled/valid/images/{name}.jpg', image)
        for cords in self.coordinates[name]:
            with open(f'./ModelDatasetled/valid/labels/{name}.txt', 'a') as file:
                file.write(
                    f'0 {str(cords)[1:-1].replace(",", " ")} \n')
            pass


# if __name__ == "__main__":
#     image = cv2.imread("../le3.png")
#     newLed = LedDetection(image, [[110, 160, 235, 380], [220, 60, 340, 280], [330, 165, 500, 350]])
#     newLed.create_dataset()
#     newLed.train_model()
