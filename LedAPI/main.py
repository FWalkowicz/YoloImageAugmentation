import os

from Detection import LedDetection
from fastapi import FastAPI, UploadFile
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import supervision as sv

storeLED = {
    'newLed': None,
    'model': YOLO(f"yolov8m.pt")
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
    print(os.getcwd())
    torch.backends.cudnn.enabled = False
    model = YOLO(f"yolov8m.pt")
    model.train(data=os.path.join(os.getcwd(), "ModelDatasetled/data.yaml"),
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

    result = YOLO(f"./runs/detect/yolo-custom/weights/best.pt").predict(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    detections_dict = {}
    detection_list = []
    for i in range(len(detections.xyxy)):
        detection = detections.xyxy[i]
        confidence = detections.confidence[i]
        detection_dict = {
            "bbox": {
                "xmin": float(detection[0]),
                "ymin": float(detection[1]),
                "xmax": float(detection[2]),
                "ymax": float(detection[3])
            },
            "confidence": float(confidence)
        }
        detection_list.append(detection_dict)

    detections_dict["detections"] = detection_list
    return detections_dict

