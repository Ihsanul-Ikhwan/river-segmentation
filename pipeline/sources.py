"""Sumber video: file (loop), webcam, RTSP/HTTP stream.

Semua source mengembalikan frame BGR via read(); eksekusi blocking
di-handle oleh CameraWorker, jadi source boleh lambat.
"""
from __future__ import annotations

import time

import cv2


class SourceError(RuntimeError):
    pass


def build_source(uri: str, loop: bool = True):
    """Factory: tentukan jenis source dari string URI.

    - angka (mis. "0" / 0)  -> webcam V4L2
    - rtsp:// http(s)://    -> stream IP camera / HLS
    - selain itu            -> file video (loop by default)
    """
    s = str(uri).strip()
    if s.isdigit():
        return WebcamSource(int(s))
    if s.startswith(("rtsp://", "rtssp://", "http://", "https://")):
        return StreamSource(s)
    return FileSource(s, loop=loop)


class FileSource:
    """Video file dengan auto-loop — dipakai untuk demo/dev.

    Di-throttle ke fps sumber supaya video ber-putar selaras waktu nyata
    (penting utk simulasi banjir yang di-bake durasinya). Tanpa pacing,
    decoder menelan 3 menit video dalam beberapa detik.
    """

    kind = "file"

    def __init__(self, path: str, loop: bool = True):
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SourceError(f"Tidak bisa buka video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        if self.fps > 120:  # sanity utk container aneh
            self.fps = 30.0
        self._next_t = time.monotonic()

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            if not self.loop:
                return None
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            self._next_t = time.monotonic()
        # pacing ke fps sumber
        self._next_t += 1.0 / self.fps
        delay = self._next_t - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.5:
            self._next_t = time.monotonic()  # jangan 'berutang' terlalu jauh
        return frame if ok else None

    def release(self):
        self.cap.release()


class WebcamSource:
    kind = "webcam"

    def __init__(self, index: int = 0, width: int = 1280, height: int = 720):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise SourceError(f"Webcam {index} tidak tersedia")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class StreamSource:
    """RTSP/HTTP stream dengan auto-reconnect.

    Buffer dibuka sekecil mungkin (BUF_SIZE=1) supaya frame selalu
    segar — penting untuk CCTV live. Transport RTSP dipaksa TCP lewat
    OPENCV_FFMPEG_CAPTURE_OPTIONS (banyak kamera Hikvision/Dahua macet
    lewat UDP di balik NAT/switch), plus timeout baca 5 detik supaya
    stream mati tidak menggantung selamanya.
    """

    kind = "stream"

    def __init__(self, url: str, reconnect_wait: float = 3.0):
        self.url = url
        self.reconnect_wait = reconnect_wait
        import os
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                              "rtsp_transport;tcp|stimeout;5000000")
        self._open()

    def _open(self):
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            raise SourceError(f"Stream tidak bisa dibuka: {self.url}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            if ok:
                return frame
            self.release()
        # reconnect
        time.sleep(self.reconnect_wait)
        self._open()
        return None

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
