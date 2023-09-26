import base64
from django.shortcuts import render
import cv2
from django.http import HttpResponse, StreamingHttpResponse
from django.views.decorators import gzip

from utils.video_camera import VideoCamera
from utils.yolo_detection import Model

# Create your views here.
@gzip.gzip_page
def webcam(request):
    return render(request, 'webcam.html')

def cam(request):
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Konversi gambar dari BGR menjadi RGB (OpenCV menggunakan format BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Ubah format frame untuk ditampilkan di halaman HTML
        ret, jpeg = cv2.imencode('.jpg', frame_rgb)
        frame_data = jpeg.tobytes()
        # cv2.imshow(cap);

        return render(request, 'cam.html', {'frame_data': frame_data})
        # return render(request, 'cam.html')

# def webcam_feed(request):
#     try:
#         cap = cv2.VideoCapture(0)  # Menggunakan kamera utama (index 0)
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             _, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     finally:
#         cap.release()

# def webcam_view(request):
#     try:
#         return StreamingHttpResponse(webcam_feed(request), content_type='multipart/x-mixed-replace; boundary=frame')
#     except GeneratorExit:
#         pass

# def stream(request):
#     # Membuka koneksi webcam
#     response = HttpResponse(content_type="multipart/x-mixed-replace;boundary=frame")
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Mengubah format gambar dari BGR ke RGB
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#         # Konversi frame menjadi format yang bisa ditampilkan di HTML
#         # ret, buffer = cv2.imencode('.jpg', frame_rgb)
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame_base64 = base64.b64encode(buffer).decode('utf-8')
#         response.write(b'--frame\r\n')
#         response.write(b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

#         context = {'frame': frame_base64}
#         return render(request, 'stream.html', context)
    
#     cap.release()
#     cv2.destroyAllWindows()

def index(request):
    return render(request, 'index.html')


def get_camera(response):
    cam = VideoCamera()
    try:
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except FileNotFoundError:
        cam.delete()


def gen(camera):
    try:
        model = Model()

        while True:
            frame = camera.get_frame()
            det = model.detect(original_image=frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + det + b'\r\n\r\n')

    except TypeError:
        camera.delete()