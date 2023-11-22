import os
from Detection import LedDetection
from fastapi import FastAPI, UploadFile, HTTPException, Depends, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from typing import Annotated, Dict
import numpy as np
import secrets
import shutil
import cv2
from ultralytics import YOLO
import torch
import supervision as sv
import threading
import json

storeLED = {"newLed": None, "model": YOLO(f"yolov8m.pt"), "isTraining": False}
UserStorage = {"clientId": "comcore", "clientSecret": "75TF3R7HrqFB"}

app = FastAPI()
security = HTTPBasic()


def get_current_username(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)]
):
    current_username_bytes = credentials.username.encode("utf8")
    correct_username_bytes = b"comcore"
    is_correct_username = secrets.compare_digest(
        current_username_bytes, correct_username_bytes
    )
    current_password_bytes = credentials.password.encode("utf8")
    correct_password_bytes = b"75TF3R7HrqFB"
    is_correct_password = secrets.compare_digest(
        current_password_bytes, correct_password_bytes
    )
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


async def check_for_image(image):
    if not image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPEG and PNG images are supported."}

    image_data = await image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image


@app.get("/auth")
def read_current_user(credentials: Annotated[str, Depends(get_current_username)]):
    """
    Get information about the current user.

    :param credentials: Current user's username.
    :return: User information.
    """
    return {"username": credentials}


@app.post("/createDataset")
async def execute_ai(input_image: UploadFile, objects):
    """
    Process an input image to create a dataset for LED detection.
    Example for coords: [[110, 160, 235, 380], [220, 60, 340, 280], [330, 165, 500, 350]]
    Example: {"detections": [{"coordinates": [110, 160, 235, 380], "label": "led1"},{"coordinates": [220, 60, 340, 280], "label": "led2"}, {"coordinates": [330, 165, 500, 350], "label": "led3"}] }

    :param input_image:  Uploaded image file (JPEG or PNG).
    :param coordinates: List of coordinates representing LED bounding boxes in the image.
    :return: Success message if the image is processed successfully.
    """
    if not input_image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPEG and PNG images are supported."}

    image_data = await input_image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    data = json.loads(objects)
    coordinates = []
    labels = []
    for obiekt in data['detections']:
        coordinates.append(obiekt['coordinates'])
        labels.append(obiekt['label'])
    newLed = LedDetection(image, data)
    storeLED["newLed"] = newLed
    newLed.create_dataset(labels)
    return {"message": "Image processed successfully"}


@app.post("/createNewModel")
async def create_new_model(name: str) -> dict[str, str]:
    """
    Start training a new LED detection model.

    :param name: Name for the new model.
    :return: Success message if the model training is started.
    """
    if not storeLED["isTraining"]:
        t = threading.Thread(target=train_model, args=(name, ))
        t.start()
        storeLED["isTraining"] = True
        return {"message": "Model training started"}
    else:
        raise HTTPException(status_code=404, detail="Model is already in training")


def train_model(name: str):
    torch.backends.cudnn.enabled = False
    model = YOLO(f"yolov8m.pt")
    model.train(
        data=os.path.join(os.getcwd(), "ModelDatasetled/data.yaml"),
        imgsz=640,
        epochs=10,
        batch=8,
        name=f"{name}",
    )
    storeLED["isTraining"] = False
    shutil.rmtree('./ModelDatasetled')


@app.get("/showModels")
def show_models():
    """
    Retrieve a list of available LED detection models.

    :return: List of model names.
    """
    models_dic = {}
    if os.path.exists("./runs/detect"):
        models_dic["models"] = os.listdir("./runs/detect")
    else:
        raise HTTPException(status_code=404, detail="Currently no model is saved")
    return models_dic


@app.post("/getPrediction")
async def get_prediction(input_image: UploadFile, model_name):
    """
    Get LED detection predictions for an input image using a specified model.

    :param input_image: Uploaded image file (JPEG or PNG).
    :param model_name: Name of the trained model.
    :return: List of dictionaries containing bounding box coordinates and confidence scores for each detected LED.
    """
    if not input_image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPEG and PNG images are supported."}

    image_data = await input_image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = YOLO(f"./runs/detect/{model_name}/weights/best.pt").predict(image)[0]
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
                "ymax": float(detection[3]),
            },
            "confidence": float(confidence),
        }
        detection_list.append(detection_dict)

    detections_dict["detections"] = detection_list
    return detections_dict


@app.get("/downloadModel")
def download_model(model_name):
    """
    Download a trained LED detection model file.

    :param model_name: Name of the model to be downloaded.
    :return: Downloaded model file in binary format.
    """
    model_dir = os.path.join("./runs/detect", f"{model_name}")
    if os.path.exists(model_dir):
        return FileResponse(
            f"./runs/detect/{model_name}/weights/best.pt",
            media_type="application/octet-stream",
            filename=f"{model_name}.pt ",
        )
    else:
        raise HTTPException(status_code=404, detail="Model not found")


@app.delete("/deleteModel")
def delete_model(model_name):
    """
    Delete a trained LED detection model.

    :param model_name: model_name: Name of the model to be downloaded.
    :return: None
    """
    model_dir = os.path.join("./runs/detect", f"{model_name}")
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    else:
        raise HTTPException(status_code=404, detail="Model not found")
