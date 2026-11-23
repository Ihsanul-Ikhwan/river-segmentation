"""Test ketahanan engine: exception detector, sumber mati, throttle callback."""
import os
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.engine import CameraConfig, CameraPipeline, load_cameras  # noqa: E402
from pipeline.detectors.base import BaseDetector, Detection              # noqa: E402
import pipeline.engine as engine_mod                                      # noqa: E402

import cv2

FRAMES = "/tmp/rw-test-frames"           # noqa: E402
engine_mod.FRAMES_DIR = FRAMES


class FlakyDetector(BaseDetector):
    """Melempar exception tiap N frame — engine harus tetap hidup."""

    name = "flaky"

    def __init__(self, every: int = 3):
        self.every = every
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        if self.calls % self.every == 0:
            raise RuntimeError("detektor meledak (sengaja)")
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[int(frame.shape[0] * 0.6):, :] = True
        return Detection(mask=mask, water_ratio=float(mask.mean()))


class AlwaysOkDetector(BaseDetector):
    name = "ok"

    def detect(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=bool)
        mask[int(frame.shape[0] * 0.6):, :] = True
        return Detection(mask=mask, water_ratio=float(mask.mean()))


def _make_video(path, seconds=2, fps=30, size=(160, 120)):
    import subprocess
    subprocess.run(
        [os.environ.get("FFMPEG", "ffmpeg"), "-y", "-f", "lavfi",
         "-i", f"testsrc=size={size[0]}x{size[1]}:rate={fps}:duration={seconds}",
         "-pix_fmt", "yuv420p", path], capture_output=True, check=True)


def test_engine_bertahan_dari_exception_detector(tmp_path, monkeypatch):
    vid = str(tmp_path / "v.mp4")
    _make_video(vid)
    cfg = CameraConfig(id="flaky-cam", name="flaky", source=vid,
                       fps=30, detector="heuristic")
    pipe = CameraPipeline(cfg)
    pipe.detector = FlakyDetector(every=3)         # injeksi setelah init
    events = []
    pipe.on_reading = lambda c, s, smp, r, ch, j: events.append(s)
    pipe.start()
    time.sleep(2.5)
    alive = pipe._thread.is_alive()
    pipe.stop()
    assert alive, "thread analisis MATI karena exception — harusnya tahan"
    assert pipe.detector.calls > 5, "detector tidak terus dipanggil setelah error"


def test_callback_dithrottle_bukan_per_frame(tmp_path):
    vid = str(tmp_path / "v.mp4")
    _make_video(vid, seconds=3)
    cfg = CameraConfig(id="throttle-cam", name="t", source=vid, fps=30)
    pipe = CameraPipeline(cfg)
    pipe.detector = AlwaysOkDetector()
    stamps = []
    pipe.on_reading = lambda c, s, smp, r, ch, j: stamps.append(time.time())
    pipe.start()
    time.sleep(4.0)
    pipe.stop()
    frames_produced = pipe.worker.frames_read
    assert frames_produced > 20
    # 4 detik → maksimal ~2-3 callback (per 2 dtk), bukan puluhan
    assert len(stamps) <= 6, f"callback {len(stamps)}x dalam 4 dtk — tidak dithrottle"


def test_status_file_ditulis_dan_valid(tmp_path):
    vid = str(tmp_path / "v.mp4")
    _make_video(vid)
    cfg = CameraConfig(id="status-cam", name="s", source=vid, fps=30)
    pipe = CameraPipeline(cfg)
    pipe.detector = AlwaysOkDetector()
    pipe.start()
    time.sleep(1.5)
    import json
    fp = os.path.join(FRAMES, "status-cam.status.json")
    assert os.path.exists(fp)
    st = json.load(open(fp))
    assert st["stage"] in ("NORMAL", "WASPADA", "SIAGA", "BAHAYA")
    assert st["stale"] is False
    assert os.path.exists(os.path.join(FRAMES, "status-cam.jpg"))
    pipe.stop()


def test_load_cameras_tolak_id_ganda(tmp_path):
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "cameras:\n"
        "  - {id: a, source: x.mp4}\n"
        "  - {id: a, source: y.mp4}\n")
    import pytest
    with pytest.raises(ValueError, match="ganda"):
        load_cameras(str(bad))
