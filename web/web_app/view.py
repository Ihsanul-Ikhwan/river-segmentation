import cv2
from django.http import HttpResponse, StreamingHttpResponse
from django.views import View
from django.template import Template, Context

from utils.yolo_detection import Model

# class WebcamView(View):
#     # template_name = 'stream.html'
#     template_name = './web/stream.html'

#     def get(self, request, *args, **kwargs):
#         return self.render_to_response({})

#     def render_to_response(self, context, **response_kwargs):
#         return self.response_class(
#             request=self.request,
#             template=self.get_template_names(),
#             context=context,
#             **response_kwargs
#         )

class VideoCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(2)  # Menggunakan kamera utama (indeks 0)
        # self.cap = cv2.VideoCapture("../sample.mp4")  # Menggunakan kamera utama (indeks 0)

    def __del__(self):
        self.cap.release()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
def webcam_feed(request):
    camera = VideoCamera()
    return StreamingHttpResponse(gen(camera), content_type="multipart/x-mixed-replace;boundary=frame")

def gen(camera):
    # while True:
    #     frame = camera.get_frame()
    #     if frame is not None:
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
    try:
        model = Model()

        while True:
            frame = camera.get_frame()
            det = model.detect(original_image=frame)

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + det + b'\r\n\r\n')

    except TypeError:
        camera.__del__()