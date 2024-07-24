from ultralytics import YOLO
import cv2
import supervision as sv

model = YOLO('/home/filip/PycharmProjects/yoloyImageGenerator/LedAPI/runs/detect/led_2/weights/best.pt')
box_annotator = sv.BoxAnnotator()
labels = ['led1', 'led2', 'led3']
image = cv2.imread('/home/filip/PycharmProjects/yoloyImageGenerator/testImages/led2.png')
image2 = cv2.resize(image, (640, 640))
#cv2.imshow('img', image)
results = model.predict(image2)[0]
detections = sv.Detections.from_ultralytics(results)
detections = detections[detections.confidence >= 0.50]

labels = [f"led{class_id + 1} {round(confidence * 100)}%" for _, _, confidence, class_id, _ in detections]
annotated_image = box_annotator.annotate(image2, detections, labels=labels)
cv2.imshow('img', annotated_image)

cv2.waitKey(0)
