#!/usr/bin/env python
"""Training YOLOv8-seg segmentasi air di DGX Spark (GB10).

Pakai:
  .venv/bin/python scripts/train_water.py                # yolov8n-seg, 80ep
  .venv/bin/python scripts/train_water.py --model yolov8s-seg.pt --epochs 120

Catatan memori GB10: engine Hermes ~75GB + vLLM ~30GB. yolov8n-seg butuh
~6-8GB — masih muat. Kalau OOM: tidur vLLM dulu
(POST http://127.0.0.1:8001/sleep?level=2).
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_YAML = os.path.join(REPO, "data", "datasets", "water", "water.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n-seg.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--name", default="water_yolov8n")
    args = ap.parse_args()

    if not os.path.exists(DATA_YAML):
        sys.exit(f"Dataset belum ada: {DATA_YAML} — jalankan build_dataset.py dulu")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        data=DATA_YAML,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=4,
        project=os.path.join(REPO, "runs", "seg"),
        name=args.name,
        patience=25,
        cos_lr=True,
        close_mosaic=12,
        cache="disk",
        plots=True,
        val=True,
    )
    # simpan best.pt ke lokasi yang dipakai config
    best = os.path.join(REPO, "runs", "seg", args.name, "weights", "best.pt")
    dst = os.path.join(REPO, "data", "weights", "water_yolov8n.pt")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(best):
        import shutil
        shutil.copy(best, dst)
        print(f"\n✔ Model terbaik tersimpan di: {dst}")


if __name__ == "__main__":
    main()
