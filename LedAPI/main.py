from Detection import LedDetection
from fastapi import FastAPI, UploadFile
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import supervision as sv

storeLED = {
    'newLed': None,
    'model': YOLO(f"../yolov8m.pt")
}

app = FastAPI()


async def check_for_image(image):
    if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPEG and PNG images are supported."}

    image_data = await image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@app.post("/createDataset")
async def execute_ai(input_image: UploadFile, coordinates):
    """
    [[110, 160, 230, 380], []]
    """
    if not input_image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPEG and PNG images are supported."}

    image_data = await input_image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    newLed = LedDetection(image, eval(coordinates))
    storeLED['newLed'] = newLed
    print(type(eval(coordinates)), eval(coordinates))
    newLed.create_dataset()

    return {"message": "Image processed successfully"}


@app.post("/createNewModel")
async def create_new_model():
    #torch.backends.cudnn.enabled = False
    model = YOLO(f"../yolov8m.pt")
    model.train(data="/home/filip/PycharmProjects/LedDetection/LedAPI/ModelDatasetled/data.yaml",
                imgsz=640,
                epochs=10,
                batch=8,
                name=f"yolo-custom")


@app.post("/getPrediction")
async def get_prediction(input_image: UploadFile):
    if not input_image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPEG and PNG images are supported."}

    image_data = await input_image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = YOLO(f"../yolov8m.pt").predict(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    return detections.xyxy
