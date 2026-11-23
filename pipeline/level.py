"""Ekstraksi muka air & konversi pixel → sentimeter.

Idea kalibrasi (praktis untuk CCTV sungai):
  - `datum_y`   : baris pixel yang dianggap level 0 cm (mis. garis patok
                  terendah / dasar sungai yang terlihat).
  - `cm_per_px` : skala vertikal (kalibrasi dengan patok meteran nyata,
                  atau dua titik diketahui).
  - Garis air per frame = baris TERATAS di mana rasio piksel air pada
                  baris itu >= `row_ratio` (difilter di ROI polygon).

Banjir bandang = kemiringan (slope) naik, jadi history disimpan per
titik waktu untuk menghitung rise-rate.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Calibration:
    datum_y: float = 0.0          # pixel row = 0 cm (0 → pakai dasar frame)
    cm_per_px: float = 1.0
    roi_polygon: list | None = None   # [(x, y), ...] area analisis (opsional)
    row_ratio: float = 0.30       # baris dianggap "air" bila >=30% piksel air
    min_water_px: int = 500       # mask terlalu kecil → anggap tidak valid

    @classmethod
    def from_dict(cls, d: dict | None) -> "Calibration":
        d = d or {}
        return cls(
            datum_y=float(d.get("datum_y", 0.0)),
            cm_per_px=float(d.get("cm_per_px", 1.0)),
            roi_polygon=d.get("roi_polygon"),
            row_ratio=float(d.get("row_ratio", 0.30)),
            min_water_px=int(d.get("min_water_px", 500)),
        )

    def roi_mask(self, shape) -> np.ndarray:
        h, w = shape[:2]
        if not self.roi_polygon:
            return np.ones((h, w), dtype=bool)
        key = (h, w)
        cached = getattr(self, "_roi_cache", None)
        if cached is None:
            cached = self._roi_cache = {}
        if key in cached:
            return cached[key]
        import cv2

        pts = np.array(self.roi_polygon, dtype=np.int32)
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(m, [pts], 1)
        out = m.astype(bool)
        cached[key] = out
        return out


@dataclass
class LevelSample:
    ts: float
    waterline_y: float | None   # None = tidak terdeteksi
    level_cm: float | None
    water_ratio: float
    above_normal_cm: float | None = None  # relatif baseline "normal"


class LevelTracker:
    """Stateful: menerima mask per frame, menghasilkan LevelSample + rise rate.

    Baseline = level "normal" yang dipelajari otomatis selama `baseline_window`
    detik pertama (median). Risk menilai tinggi air RELATIF terhadap baseline,
    sehingga kamera tanpa kalibrasi patok pun tetap masuk akal: "air naik X cm
    dari normal". Kamera berpatok bisa pakai level absolut via calib.datum_y.
    """

    def __init__(self, calib: Calibration, window_sec: float = 600.0,
                 median_span: int = 5, baseline_window: float = 45.0,
                 relative_level: bool = True):
        self.calib = calib
        self.window_sec = window_sec
        self.median_span = median_span
        self.baseline_window = baseline_window
        self.relative_level = relative_level
        self._samples: deque[LevelSample] = deque()
        self._recent_lines: deque[float] = deque(maxlen=median_span)
        self._baseline_buf: list[float] = []
        self._baseline: float | None = None
        self._t0: float | None = None

    def update(self, mask: np.ndarray | None, ts: float | None = None) -> LevelSample:
        ts = ts or time.time()
        if self._t0 is None:
            self._t0 = ts
        waterline_y = None
        level_cm = None

        if mask is not None and mask.any():
            roi = self.calib.roi_mask(mask.shape)
            m = mask & roi
            if int(m.sum()) >= self.calib.min_water_px:
                w_ratio_of_row = m.mean(axis=1)  # rasio air per baris
                rows = np.where(w_ratio_of_row >= self.calib.row_ratio)[0]
                if len(rows):
                    waterline_y = float(rows.min())

        if waterline_y is not None:
            self._recent_lines.append(waterline_y)
            # median smoothing anti-kedip riak
            y = float(np.median(self._recent_lines))
            level_cm = (self.calib.datum_y - y) * self.calib.cm_per_px \
                if self.calib.datum_y else (mask.shape[0] - y) * self.calib.cm_per_px

        sample = LevelSample(ts=ts, waterline_y=waterline_y,
                             level_cm=level_cm,
                             water_ratio=float(mask.mean()) if mask is not None else 0.0,
                             above_normal_cm=self.level_above_baseline(level_cm))
        self._samples.append(sample)
        self._trim(ts)
        self._learn_baseline(level_cm, ts)
        return sample

    def _learn_baseline(self, level_cm: float | None, ts: float):
        if level_cm is None or self._t0 is None:
            return
        if ts - self._t0 <= self.baseline_window:
            self._baseline_buf.append(level_cm)
            self._baseline = float(np.median(self._baseline_buf))
        elif self._baseline is None:
            self._baseline = level_cm  # fallback bila window terlewat

    def level_above_baseline(self, level_cm: float | None) -> float | None:
        if level_cm is None or self._baseline is None:
            return None
        return level_cm - self._baseline

    @property
    def baseline_cm(self) -> float | None:
        return self._baseline

    def _trim(self, now: float):
        cutoff = now - self.window_sec - 5
        while self._samples and self._samples[0].ts < cutoff:
            self._samples.popleft()

    def rise_rate_cm_per_min(self, lookback_sec: float = 120.0) -> float | None:
        """Slope level air (cm/menit) via least-squares pada window lookback.

        Acuan 'now' = timestamp SAMPEL TERAKHIR (bukan jam dinding) supaya
        konsisten utk sumber dengan ts offline/replay.
        """
        if not self._samples:
            return None
        now = self._samples[-1].ts
        pts = [(s.ts, s.level_cm) for s in self._samples
               if s.level_cm is not None and s.ts >= now - lookback_sec]
        if len(pts) < 3:
            return None
        ts = np.array([p[0] for p in pts])
        lv = np.array([p[1] for p in pts])
        if ts[-1] - ts[0] < 10:  # belum cukup rentang waktu
            return None
        slope_per_sec = np.polyfit(ts, lv, 1)[0]
        return float(slope_per_sec * 60.0)

    def history(self) -> list[LevelSample]:
        return list(self._samples)
