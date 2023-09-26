import time
from ultralytics import YOLO
import cv2

# Load a pretrained YOLOv8n model
model = YOLO('weight/best.pt')

# Open the video file
video_path = "path/to/your/video/file.mp4"
cap = cv2.VideoCapture(2)

ms = 0
        
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        for r in results:
            
            res = r.boxes.xyxy.cpu().numpy().astype(int)
            if (len(res) == 0):
                continue
            
            if (len(res[0]) == 0):
                continue
            print("CALIAK SIKO", res[0])
            speed = r.speed['preprocess'] + r.speed['inference'] + r.speed['postprocess']
            panjang = (-1 * res[0][1]) + 332
            
            if (panjang <= 0):
                ms = 0
            else:    
                ms += speed    
            
            
            spd_str = "{:.2f}".format(panjang/ms*1000)
            est = "{:.2f}".format((332-panjang) / (panjang/ms*1000))
            ms_str = "{:.0f}".format(ms // 1000)
                            
            # print("JAN CALIAK SIKO", r.speed)
            annotated_frame = cv2.line(frame, (347, 332) ,(347, 0),  (0,0,255),2)
            annotated_frame = cv2.line(annotated_frame, (347, 332), (347, res[0][1]),  (0,255,0),2)
            annotated_frame = cv2.rectangle(annotated_frame, (res[0][0], res[0][1]),(res[0][2], res[0][3]),  (0,0,255),2)
             
            annotated_frame = cv2.putText(annotated_frame, f'Panjang = {panjang} px', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            annotated_frame = cv2.putText(annotated_frame, f'total {ms_str} s', (50,100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            annotated_frame = cv2.putText(annotated_frame, f'speed {spd_str} px/s', (50,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            annotated_frame = cv2.putText(annotated_frame, f'est {est} s', (50,200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            # annotated_frame = cv2.putText(annotated_frame, f'{speed} ms', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 1)
            
        
        # Visualize the results on the frame
        # annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()