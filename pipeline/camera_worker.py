"""CameraWorker: satu thread per kamera, pola 'latest frame'.

Reader terus menarik frame dari source; konsumen (analyzer / streamer)
mengambil frame terbaru tanpa blocking. Kalau analyzer lambat, frame
buangan — bukan antre membludak. Ini pola standar untuk CCTV realtime.
"""
from __future__ import annotations

import threading
import time

from .sources import build_source


class CameraWorker:
    def __init__(self, camera_id: str, uri: str, loop: bool = True, name: str = ""):
        self.camera_id = camera_id
        self.name = name or camera_id
        self._uri = uri
        self._loop = loop
        self._lock = threading.Lock()
        self._frame = None
        self._frame_seq = 0
        self._frame_ts = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.frames_read = 0

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"cam-{self.camera_id}", daemon=True
        )
        self._thread.start()

    def _run(self):
        source = None
        while not self._stop.is_set():
            try:
                if source is None:
                    source = build_source(self._uri, loop=self._loop)
                frame = source.read()
                if frame is None:
                    self.error = f"sumber '{self._uri}' EOF/reconnect"
                    time.sleep(0.05)
                    continue
                self.error = None
                with self._lock:
                    self._frame = frame
                    self._frame_seq += 1
                    self._frame_ts = time.time()
                self.frames_read += 1
            except Exception as exc:  # noqa: BLE001 — worker tidak boleh mati
                self.error = str(exc)
                if source is not None:
                    try:
                        source.release()
                    except Exception:
                        pass
                    source = None
                time.sleep(1.0)
        if source is not None:
            try:
                source.release()
            except Exception:
                pass

    def get_frame(self):
        """Ambil (frame, seq). frame=None kalau belum ada."""
        with self._lock:
            return (None if self._frame is None else self._frame.copy(),
                    self._frame_seq, self._frame_ts)

    def stop(self, timeout: float = 2.0):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())
