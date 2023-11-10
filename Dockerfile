## Obraz bazowy
FROM python:3.9-bullseye

#install pip dependencies
COPY requirements.txt .
RUN pip3 install -r ./requirements.txt

## Instalacja paczek
RUN apt-get update
RUN apt update && apt install -y libsm6 libxext6 ffmpeg libfontconfig1 libxrender1 libgl1-mesa-glx

COPY LedAPI .



