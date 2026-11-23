#!/usr/bin/env python
"""Probe cepat: heuristik detektor air di video sampel.

Pakai: .venv/bin/python scripts/probe_detector.py [video.mp4] [--save N]
Simpan frame ber-annotasi ke scratch untuk dicek pakai mata.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from pipeline.detectors import get_detector
from pipeline.level import Calibration, LevelTracker

def main():
    video = sys.argv[1] if len(sys.argv) > 1 else "data/samples/bahorok_sumatra.webm"
    det = get_detector("heuristic")
    tracker = LevelTracker(Calibration.from_dict({}))
    cap = cv2.VideoCapture(video)
    n = 0
    out_dir = os.environ.get("OUT_DIR", ".")
    os.makedirs(out_dir, exist_ok=True)
    save_every = int(os.environ.get("SAVE_EVERY", "60"))
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        d = det.detect(frame)
        s = tracker.update(d.mask)
        if n % 15 == 0:
            lv = "—" if s.level_cm is None else f"{s.level_cm:.0f}cm"
            print(f"f{n:05d} ratio={d.water_ratio:.2f} "
                  f"waterline={s.waterline_y} level={lv}")
        if n % save_every == 1:
            ann = det.annotate(frame.copy(), d)
            if s.waterline_y:
                cv2.line(ann, (0, int(s.waterline_y)),
                         (ann.shape[1], int(s.waterline_y)), (0, 255, 255), 2)
            cv2.imwrite(os.path.join(out_dir, f"probe_{os.path.basename(video)}_{n:05d}.jpg"), ann)
    cap.release()
    print("total frames:", n)

if __name__ == "__main__":
    main()
