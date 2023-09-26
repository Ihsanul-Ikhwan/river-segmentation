import os

import cv2
from ultralytics import YOLO


class Model:
    """Model class"""
    weight_name = "best.pt"
    # weight_name = "newbest.pt"
    # weight_name = "yolov8x-seg.pt"
    model = None
    
    def __init__(self):
        for f in os.listdir('web/utils/weight'):
            # print("data",f)
            if f == self.weight_name:
                self.model = YOLO(f'web/utils/weight/{f}')
            else:
                print("Model is not found!")
        # self.model = YOLO(self.weight_name)

    def detect(self, original_image):
        # Load Model
        
        # for f in os.listdir('web/utils/weight'):
        #     if f == self.weight_name:
        #         self.model = YOLO(f'web/utils/weight/{f}')
        #     else:
        #         print("Model is not found!")
        
        if self.model is None:
            raise Exception("NO MODEL!")
        else:
            # print("Model is", self.model)
            print("Model is")

        try:
            # Run inference from input
            results = self.model(original_image, device=0, show=False)
            
            something = self.calculate(results)
            print("SINI WOE!!! ",something)
            
            # Visualize results on the frame
            annotatedframe = results[0].plot()

            # return as image
            _, jpg = cv2.imencode('.jpg', annotatedframe)

            return jpg.tobytes()
        except TypeError as e:
            print("Error: ", e)
        
    def calculate(self, results):
        try:
            for result in results:
                # for label, confidence, bbox in result.pred:
                #     # Dapatkan koordinat x, y, width, dan height dari bounding box
                #     x, y, width, height = bbox.xyxy().flatten()

                #     # Hitung titik tengah pada garis atas bounding box
                #     x_center = x + (width / 2)
                #     y_center = y

                #     # Tambahkan titik tengah ke dalam list
                #     centers.append((label, x_center, y_center))

            # Kembalikan list dengan titik tengah
                return result.boxes.xywh()
        except TypeError as e:
            print("Error: ", e) 
