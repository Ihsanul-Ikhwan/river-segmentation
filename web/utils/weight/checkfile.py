import os
from ultralytics import YOLO


def check_file(model_name):
    """
    returns YOLO model from specified location.
            Parameters:
                model_name (string): model's name
    """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for f in os.listdir(current_path):
        if f == model_name:
            return YOLO(f'{current_path}/{model_name}')
    return None
