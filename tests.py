import cv2

# TODO: 1. rotate, 2. flip, 3. crop
# TODO: minimum 15 zdjec

# TODO: Zrobienie struktury dataset'u dla yolo
# TODO: Generowanie pliku .yaml do trenowania modelu

image = cv2.imread('led.jpg')
flip = cv2.flip(image, 1)
rotate = cv2.rotate(image, 1)

while True:
    cv2.imshow('image', image)
    cv2.imshow('flip', flip)
    cv2.imshow('rotate', rotate)
    if cv2.waitKey(1) == ord('q'):
        break

