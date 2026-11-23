"""Detektor air heuristik (tanpa model) — untuk bootstrap demo & pseudo-label.

Kombinasi:
  1. Rentang HSV air keruh (biru-hijau-abu gelap, saturasi rendah-sedang)
  2. Top-hat tekstur: permukaan air punya pola riak khas (frekuensi tinggi,
     amplitudo rendah) — dihitung via Gaussian blur difference, threshold.
  3. Morfologi + ambil connected component terbesar, lalu isi (fill) ke bawah
     dari garis air supaya badan sungai jadi satu massa.

Akurasi tidak akan semodel YOLO, tapi cukup untuk demo pipeline dan untuk
membuat pseudo-label awal saat training dataset pertama.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import BaseDetector, Detection


class HeuristicWaterDetector(BaseDetector):
    name = "heuristic"

    def __init__(self,
                 hsv_low=(0, 0, 20),
                 hsv_high=(180, 170, 210),
                 min_area_frac: float = 0.02,
                 fill_below: bool = True):
        self.hsv_low = np.array(hsv_low, dtype=np.uint8)
        self.hsv_high = np.array(hsv_high, dtype=np.uint8)
        self.min_area_frac = min_area_frac
        self.fill_below = fill_below

    def detect(self, frame_bgr: np.ndarray) -> Detection:
        h, w = frame_bgr.shape[:2]
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # 1) filter warna air
        color_mask = cv2.inRange(hsv, self.hsv_low, self.hsv_high)

        # 2) energi riak: |gray - blur| menandai tekstur halus.
        #    Seed = irisan warna-air DAN riak (bukan union — union dulu
        #    bikin seluruh frame dianggap air karena vegetasi ikut kena).
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (15, 15), 0)
        ripple = cv2.absdiff(gray, blur)
        _, ripple_mask = cv2.threshold(ripple, 6, 255, cv2.THRESH_BINARY)
        ripple_d = cv2.dilate(ripple_mask,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

        mask = cv2.bitwise_and(color_mask, ripple_d)

        # 3) morfologi: tutup lubang & buang noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

        # 4) ambil komponen terbesar
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n <= 1:
            return Detection(mask=np.zeros((h, w), dtype=bool), water_ratio=0.0)
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = int(np.argmax(areas)) + 1
        comp = (labels == biggest).astype(np.uint8) * 255

        if comp.sum() / 255 < self.min_area_frac * h * w:
            return Detection(mask=np.zeros((h, w), dtype=bool), water_ratio=0.0)

        # 5) fill-below: dari garis air tiap kolom, anggap semua ke bawah air.
        #    Menyamarkan refleksi/pantulan di tengah sungai.
        if self.fill_below:
            ys = np.argmax(comp > 0, axis=0)  # y teratas per kolom (0 klo kosong)
            has = (comp > 0).any(axis=0)
            cols = np.arange(w)[None, :]
            rows = np.arange(h)[:, None]
            comp = np.where(has[None, :] & (rows >= ys[None, :]), 255, 0).astype(np.uint8)

        bool_mask = comp > 0
        return Detection(mask=bool_mask, water_ratio=float(bool_mask.mean()))
