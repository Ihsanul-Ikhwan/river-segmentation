import os
from ultralytics import YOLO

weight_name = "yolov8x-seg.pt"
a = None
model = None

# for f in os.listdir('web/utils/weight'):
#     print(f)
    
for f in os.listdir('web/utils/weight'):
    if f == weight_name:
        # model = YOLO(f'web/utils/weight/{f}')
        a = weight_name
    else:
        print("Model is not found!")
print(a)