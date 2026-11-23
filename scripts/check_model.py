#!/usr/bin/env python
"""Cek cepat model air di satu video: simpan frame ber-annotasi."""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import cv2

from pipeline.detectors import get_detector


def main():
    video = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("TMPDIR", "/tmp")
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    weight = os.path.join(REPO, "data", "weights", "water_yolov8n.pt")
    det = get_detector("yolo", weight_path=weight)
    cap = cv2.VideoCapture(video)
    n, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        if n % every == 1:
            d = det.detect(frame)
            ann = det.annotate(frame.copy(), d)
            cv2.putText(ann, f"water {d.water_ratio:.0%} conf {d.extra.get('mean_conf', 0):.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(os.path.join(out_dir, f"yolo_check_{n:04d}.jpg"), ann)
            saved += 1
    print(f"cek {saved} frame dari {n} — output di {out_dir}")


if __name__ == "__main__":
    main()
