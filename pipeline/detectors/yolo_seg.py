"""Detektor air berbasis YOLOv8-seg (ultralytics).

Model dilatih dengan satu kelas: `water` (polygon segmentasi).
Kalau weight belum ada, bisa fallback ke class `water` pada dataset COCO? —
tidak ada, jadi weight hasil training WAJIB. Engine akan fallback ke
heuristic bila file tidak ditemukan.
"""
from __future__ import annotations

import os

import numpy as np

from .base import BaseDetector, Detection


class YoloWaterDetector(BaseDetector):
    name = "yolo"

    def __init__(self, weight_path: str, device: str | None = None,
                 conf: float = 0.35, imgsz: int = 640, water_classes=None):
        from ultralytics import YOLO  # import lambat, hanya saat dipakai

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Weight tidak ditemukan: {weight_path}")
        self.model = YOLO(weight_path)
        # auto-pilih GPU kalau ada
        if device is None:
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.conf = conf
        self.imgsz = imgsz
        # default: semua class yang namanya 'water' (bisa multi-class ke depan)
        self.water_classes = water_classes
        self._class_index: dict[int, str] | None = None

    def _is_water(self, cls_id: int, names: dict) -> bool:
        if self.water_classes is not None:
            return cls_id in self.water_classes
        name = str(names.get(cls_id, "")).lower()
        return "water" in name or name in ("air", "sungai", "banjir")

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        h, w = frame_bgr.shape[:2]
        results = self.model.predict(
            frame_bgr, device=self.device, conf=self.conf,
            imgsz=self.imgsz, verbose=False,
        )
        mask = np.zeros((h, w), dtype=bool)
        confs: list[float] = []
        for r in results:
            names = r.names if hasattr(r, "names") else {}
            if r.masks is None:
                continue
            for i, m in enumerate(r.masks.data.cpu().numpy()):
                cls_id = int(r.boxes.cls[i].item())
                if not self._is_water(cls_id, names):
                    continue
                confs.append(float(r.boxes.conf[i].item()))
                mm = cv2_resize(m, (w, h))
                mask |= mm > 0.5
        ratio = float(mask.mean())
        extra = {"mean_conf": (sum(confs) / len(confs)) if confs else 0.0,
                 "instances": len(confs)}
        return Detection(mask=mask, water_ratio=ratio, extra=extra)


def cv2_resize(arr, size):
    import cv2

    return cv2.resize(arr.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
