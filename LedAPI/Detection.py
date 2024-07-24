import albumentations as A
import os
import cv2
import random
import string
import math

class LedDetection:
    def __init__(self, first_image, image_data):
        self.first_image = cv2.resize(first_image, (640, 640))
        self.image_data = image_data
        self.transformed_image_data = None
        self.labels = []
        self.name = 'led'

    def transform_image_data(self) -> None:
        temp_data = []
        for detection in self.image_data['detections']:
            x_center, y_center, width, height = detection["coordinates"]
            object_with_label = [x_center, y_center, width, height, detection['label']]
            temp_data.append(object_with_label)

        self.transformed_image_data = temp_data

    def normalize_coordinates(self) -> None:
        normalized_data = []
        image_width, image_height = 640, 640

        for detection_list in self.transformed_image_data:
            x1, y1, width, height, label = detection_list
            normalize_x = (abs((x1 + width) / 2) / image_width)
            normalize_y = (abs((y1 + height) / 2) / image_height)
            normalize_width = abs((width - x1) / image_width)
            normalize_height = abs((height - y1) / image_height)
            data = [normalize_x, normalize_y, normalize_width, normalize_height, label]
            normalized_data.append(data)

        self.transformed_image_data = normalized_data
        print(self.transformed_image_data)

    def data_augmentation(self):
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.4, contrast_limit=0.2, p=1.0),
            A.GaussianBlur(p=0.3),
            A.ISONoise(p=0.6)
        ], bbox_params=A.BboxParams(format='yolo'))
        transformed = []
        for i in range(30):
            transformed.append(transform(image=self.first_image, bboxes=self.transformed_image_data))

        return transformed

    def create_dataset_dependencies(self):
        path = f"./ModelDataset{self.name}"
        folders = ["train", "test", "valid"]
        image_and_label_folders = ["images", "labels"]
        for data in self.transformed_image_data:
            self.labels.append(data[4])
        print(self.labels)
        configuration_file = "data.yaml"
        yaml_data = f"""train: ./train/images
val: ./valid/images
test: ./test/images

nc: {len(self.transformed_image_data)}
names: {self.labels}
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

    def cut_dataset(self, dataset):
        random.shuffle(dataset)
        split_index = math.ceil(0.8 * len(dataset))
        training_set = dataset[:split_index]
        validation_set = dataset[split_index:]
        return training_set, validation_set

    def save_images_to_dataset(self, images, folder='train'):
        for i in images:
            file_name = ''.join(random.choice(string.ascii_letters) for i in range(10))
            path_images = os.path.join(f'./ModelDatasetled/{folder}/images', file_name)
            path_labels = os.path.join(f'./ModelDatasetled/{folder}/labels', file_name)
            cv2.imwrite(f"{path_images}.jpg", i['image'])
            with open(f"{path_labels}.txt", 'w') as file:
                for coordinates in i['bboxes']:
                    file.write(
                        f"{self.labels.index(coordinates[4])} {coordinates[0]} {coordinates[1]} {coordinates[2]} {coordinates[3]}\n")

    def create_dataset(self):
        self.transform_image_data()
        self.normalize_coordinates()
        new_images = self.data_augmentation()
        self.create_dataset_dependencies()
        training_set, validation_set = self.cut_dataset(new_images)
        self.save_images_to_dataset(training_set, folder='train')
        self.save_images_to_dataset(validation_set, folder='valid')


input_data = {"detections": [
    {"coordinates": [110, 160, 235, 380], "label": "led1"},
    {"coordinates": [220, 60, 340, 280], "label": "led2"},
    {"coordinates": [330, 165, 500, 350], "label": "led3"}
]}

if __name__ == "__main__":
    image = cv2.imread("../testImages/led.png")
    newLed = LedDetection(image, input_data)
    newLed.create_dataset()

