"""Detektor air: interface + registry."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Detection:
    """Hasil deteksi air satu frame.

    mask       : HxW bool — True = piksel air
    water_ratio: framenya berapa bagian yang air (0..1)
    extra      : info bebas (conf rata-rata, jumlah instance, dst.)
    """

    mask: np.ndarray | None
    water_ratio: float = 0.0
    extra: dict = field(default_factory=dict)


class BaseDetector(ABC):
    name = "base"

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> Detection:
        raise NotImplementedError

    def annotate(self, frame_bgr: np.ndarray, det: Detection):
        """Overlay default: mask air semi-transparan cyan."""
        if det.mask is not None:
            overlay = frame_bgr.copy()
            overlay[det.mask] = (255, 200, 0)  # cyan-ish BGR
            frame_bgr = cv2_blend(frame_bgr, overlay, alpha=0.35)
        return frame_bgr


def cv2_blend(base, overlay, alpha=0.4):
    import cv2

    return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)


def get_detector(kind: str, **kwargs) -> BaseDetector:
    kind = (kind or "heuristic").lower()
    if kind == "heuristic":
        from .heuristic import HeuristicWaterDetector

        return HeuristicWaterDetector(**kwargs)
    if kind == "yolo":
        from .yolo_seg import YoloWaterDetector

        return YoloWaterDetector(**kwargs)
    raise ValueError(f"Detector tidak dikenal: {kind}")
