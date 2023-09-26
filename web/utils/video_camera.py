import threading

import cv2


class VideoCamera(object):
    def init(self):
        self.video = cv2.VideoCapture(1)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def delete(self):
        self.video.release()

    def get_frame(self):
        return self.frame

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()
