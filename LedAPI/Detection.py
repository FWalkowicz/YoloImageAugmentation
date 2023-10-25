from abc import ABC, abstractmethod
import os
import cv2
from ultralytics import YOLO
import torch


class Detection(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
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
        self.coordinates = coordinates
        self.name = "led"

    def train(self):
        torch.backends.cudnn.enabled = False
        model = YOLO(f"yolov8m.pt")
        model.train(
            data='/home/filip/PycharmProjects/LedDetection/LedDS/data.yaml',
            imgsz=640,
            epochs=10,
            batch=8,
            name=f"yolo-custom",
        )

    def create_dataset(self):
        operations = ["basic", "flip", "rotate", "blur", "noise", "grayscale", "saturation"]
        self.create_dataset_folder()
        self.normalize_coordinates()
        for operation in operations:
            print(operation)

    def predict(self):
        pass

    def normalize_coordinates(self) -> None:
        for coordinate in self.coordinates:
            x1, y1 = self.coordinates[coordinate][0], self.coordinates[coordinate][1]
            x2, y2 = self.coordinates[coordinate][2], self.coordinates[coordinate][3]
            image_width, image_height = 640, 640
            normalize_x = abs((x1 + x2) / (2 * image_width))
            normalize_y = abs((y1 + y2) / (2 * image_height))
            normalize_width = abs((x2 - x1) / image_width)
            normalize_height = abs((y2 - y1) / image_height)
            self.coordinates[coordinate] = [normalize_x, normalize_y, normalize_width, normalize_height]

    def create_dataset_folder(self):
        path = f"./ModelDataset{self.name}"
        folders = ['train', 'test', 'valid']
        image_and_label_folders = ['images', 'labels']
        configuration_file = 'data.yaml'
        yaml_data = """train: ../train/images
        val: ../valid/images
        test: ../test/images

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
            with open(os.path.join(path, configuration_file), 'w') as f:
                f.write(yaml_data)


if __name__ == "__main__":
    image = cv2.imread("../led.jpg")
    newLed = LedDetection(image, {"basic": [15, 75, 120, 200]})
    newLed.create_dataset()
