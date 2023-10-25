from Detection import LedDetection
from fastapi import FastAPI, UploadFile
import numpy as np
import cv2


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/executeAI")
async def execute_ai(input_image: UploadFile):
    """

    """
    if not input_image.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return {"error": "Only JPEG and PNG images are supported."}

    image_data = await input_image.read()
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    led = LedDetection(image, {"basic": [15, 75, 120, 200]})
    led.create_dataset()

    return {"message": "Image processed successfully"}

