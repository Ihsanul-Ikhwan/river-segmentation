"""Engine: orkestrator seluruh pipeline.

Untuk tiap kamera di config:
  CameraWorker (thread baca) → loop analisis (fps dibatasi)
    → detector (mask air) → LevelTracker (level cm) → RiskAssessor (stage)
    → callback on_reading / on_alert (disambung ke DB & Telegram oleh
      management command; engine sendiri tidak tahu Django).

Frame ber-annotasi + status JSON ditulis ke data/frames/ supaya proses
web Django bisa men-stream-nya tanpa proses shared memory.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field

import cv2
import yaml

from .camera_worker import CameraWorker
from .detectors import get_detector
from .level import Calibration, LevelTracker
from .risk import STAGE_EMOJI, STAGE_NAME, RiskAssessor, RiskConfig

log = logging.getLogger("riverwatch.engine")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRAMES_DIR = os.path.join(REPO_ROOT, "data", "frames")


@dataclass
class CameraConfig:
    id: str
    name: str
    source: str
    loop: bool = True
    fps: float = 5.0                 # batas fps analisis
    detector: str = "heuristic"
    detector_opts: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)
    record: bool = False
    notify: bool = True

    @classmethod
    def from_dict(cls, d: dict) -> "CameraConfig":
        return cls(
            id=str(d["id"]),
            name=d.get("name", str(d["id"])),
            source=str(d["source"]),
            loop=bool(d.get("loop", True)),
            fps=float(d.get("fps", 5.0)),
            detector=d.get("detector", "heuristic"),
            detector_opts=dict(d.get("detector_opts", {})),
            calibration=dict(d.get("calibration", {})),
            risk=dict(d.get("risk", {})),
            record=bool(d.get("record", False)),
            notify=bool(d.get("notify", True)),
        )


def load_cameras(path: str) -> list[CameraConfig]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cams = []
    for c in data.get("cameras", []):
        cfg = CameraConfig.from_dict(c)
        src = cfg.source
        if (not src.isdigit()
                and not src.startswith(("rtsp://", "rtmp://", "http://", "https://"))
                and not os.path.isabs(src)):
            cfg.source = os.path.join(REPO_ROOT, src)  # relatif ke repo
        wp = cfg.detector_opts.get("weight_path")
        if wp and not os.path.isabs(wp):
            cfg.detector_opts["weight_path"] = os.path.join(REPO_ROOT, wp)
        cams.append(cfg)
    if not cams:
        raise ValueError(f"Tidak ada kamera di {path}")
    return cams


class CameraPipeline:
    """Satu kamera: worker + analisis loop."""

    def __init__(self, cfg: CameraConfig, on_reading=None, on_alert=None):
        self.cfg = cfg
        self.on_reading = on_reading
        self.on_alert = on_alert
        self.worker = CameraWorker(cfg.id, cfg.source, loop=cfg.loop, name=cfg.name)
        self.calib = Calibration.from_dict(cfg.calibration)
        self.tracker = LevelTracker(
            self.calib,
            relative_level=bool(cfg.calibration.get("relative_level", True)),
            baseline_window=float(cfg.calibration.get("baseline_window", 45.0)))
        self.assessor = RiskAssessor(RiskConfig.from_dict(cfg.risk))
        self.detector = self._build_detector()
        self.status: dict = {"stage": "INIT", "level_cm": None,
                             "rise": None, "fps": 0.0, "error": None}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_seq = -1
        os.makedirs(FRAMES_DIR, exist_ok=True)

    def _build_detector(self):
        try:
            return get_detector(self.cfg.detector, **self.cfg.detector_opts)
        except (FileNotFoundError, ImportError) as exc:
            log.warning("Detektor '%s' gagal (%s) → fallback heuristik",
                        self.cfg.detector, exc)
            return get_detector("heuristic")

    def start(self):
        self.worker.start()
        self._thread = threading.Thread(target=self._run,
                                        name=f"pipe-{self.cfg.id}", daemon=True)
        self._thread.start()

    def _run(self):
        interval = 1.0 / max(self.cfg.fps, 0.1)
        fps_window: list[float] = []
        while not self._stop.is_set():
            t0 = time.time()
            frame, seq, ts = self.worker.get_frame()
            if frame is None or seq == self._last_seq:
                time.sleep(0.02)
                continue
            self._last_seq = seq

            det = self.detector.detect(frame)
            # sanity gate: mask menutupi hampir seluruh frame = detektor
            # kemungkinan blunder (langit/refleksi/heuristik lepas) → anggap
            # tidak valid daripada memancing false alarm.
            if det.mask is not None and det.water_ratio > 0.93:
                det.mask = None
                det.water_ratio = 0.0
                det.extra["rejected"] = True
            sample = self.tracker.update(det.mask, ts=ts)
            rise = self.tracker.rise_rate_cm_per_min()
            # Risk menilai tinggi air RELATIF baseline "normal" yang
            # dipelajari otomatis (kecuali kamera dikalibrasi patok absolut,
            # set relative_level=false di calibration).
            rel = self.tracker.level_above_baseline(sample.level_cm) \
                if self.tracker.relative_level else sample.level_cm
            stage, changed = self.assessor.assess(rel, rise, ts=ts)

            annotated = self._annotate(frame, det, sample, rise, stage)
            self._publish(annotated, stage, sample, rise)

            if self.on_reading:
                self.on_reading(self.cfg, stage, sample, rise, changed,
                                self._jpeg(annotated))

            dt = time.time() - t0
            fps_window.append(dt)
            if len(fps_window) > 30:
                fps_window.pop(0)
            avg = sum(fps_window) / len(fps_window)
            self.status = {
                "stage": STAGE_NAME[stage],
                "stage_code": stage,
                "level_cm": sample.level_cm,
                "level_above_normal": None if rel is None else round(rel, 1),
                "baseline_cm": self.tracker._baseline,
                "rise": rise,
                "water_ratio": round(sample.water_ratio, 4),
                "fps": round(1.0 / avg, 2) if avg else 0.0,
                "detector": self.detector.name,
                "error": self.worker.error,
                "updated": ts,
            }
            if dt < interval:
                time.sleep(interval - dt)

    def _annotate(self, frame, det, sample, rise, stage):
        out = self.detector.annotate(frame.copy(), det)
        color = {0: (80, 200, 80), 1: (0, 215, 255), 2: (0, 140, 255),
                 3: (0, 0, 255)}[stage]
        cv2.rectangle(out, (0, 0), (out.shape[1], 46), (0, 0, 0), -1)
        lv = "—" if sample.level_cm is None else f"{sample.level_cm:.0f} cm"
        rr = "—" if rise is None else f"{rise:+.1f} cm/mnt"
        cv2.putText(out, f"{STAGE_NAME[stage]}  |  level {lv}  |  {rr}",
                    (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        if sample.waterline_y is not None:
            y = int(sample.waterline_y)
            cv2.line(out, (0, y), (out.shape[1], y), color, 2)
        return out

    def _jpeg(self, frame, quality: int = 78) -> bytes:
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else b""

    def _publish(self, annotated, stage, sample, rise):
        try:
            jpeg = self._jpeg(annotated)
            if jpeg:
                tmp = os.path.join(FRAMES_DIR, f"{self.cfg.id}.jpg.tmp")
                dst = os.path.join(FRAMES_DIR, f"{self.cfg.id}.jpg")
                with open(tmp, "wb") as f:
                    f.write(jpeg)
                os.replace(tmp, dst)  # atomic → streamer tak baca separuh
            status = dict(self.status)
            status.update({"stage_code": stage,
                           "level_cm": sample.level_cm, "rise": rise})
            with open(os.path.join(FRAMES_DIR, f"{self.cfg.id}.status.json"),
                      "w", encoding="utf-8") as f:
                json.dump(status, f)
        except OSError as exc:
            log.debug("publish frame gagal: %s", exc)

    def stop(self):
        self._stop.set()
        self.worker.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)


class Engine:
    """Kumpulan CameraPipeline, hidup di proses monitor (bukan web)."""

    def __init__(self, cameras: list[CameraConfig], on_reading=None,
                 on_alert=None):
        self.pipelines = [CameraPipeline(c, on_reading=on_reading,
                                         on_alert=on_alert)
                          for c in cameras]

    def start(self):
        for p in self.pipelines:
            p.start()
            log.info("Kamera %s (%s) mulai — source=%s detektor=%s",
                     p.cfg.id, p.cfg.name, p.cfg.source, p.detector.name)

    def stop(self):
        for p in self.pipelines:
            p.stop()

    def status(self) -> list[dict]:
        return [{"id": p.cfg.id, "name": p.cfg.name,
                 "source": p.cfg.source, **p.status} for p in self.pipelines]
