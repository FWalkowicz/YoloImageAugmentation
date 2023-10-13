from abc import ABC, abstractmethod


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
    def predict(self):
        pass


class LedDetection(Detection):
    def __init__(self):
        pass

    def train(self):
        pass

    def create_dataset(self):
        pass

    def predict(self):
        pass

