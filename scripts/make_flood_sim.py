#!/usr/bin/env python
"""Buat video simulasi banjir bandang dari video sungai biasa.

Mengapa sintetis? Untuk MENGUJI pipeline peringatan secara deterministik:
level air harus naik dari normal → WASPADA → SIAGA → BAHAYA dalam beberapa
menit, lengkap dengan perubahan rise-rate, tanpa harus nunggu hujan beneran.

Caranya: tiap frame, bagian bawah gambar (di bawah garis air awal) digeser
ke atas secara bertahap (waterline naik) + tint cokelat keruh + riak statis
yang bergerak — cukup menipu detektor & mata utk demo.

Pakai:
  .venv/bin/python scripts/make_flood_sim.py data/samples/klamath.webm \
      data/samples/flood_sim_klamath.mp4 --rise-minutes 6 --speedup 12
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--rise-minutes", type=float, default=6.0,
                    help="durasi simulasi (menit TAYANG sebelum max rise)")
    ap.add_argument("--max-rise-px", type=float, default=220.0)
    ap.add_argument("--turbidity", type=float, default=0.55)
    ap.add_argument("--out-fps", type=float, default=0,
                    help="fps output; 0 = sama dgn sumber. Pakai 2-4 utk "
                         "memperpanjang tayang tanpa fake frames")
    ap.add_argument("--dump-masks", default="",
                    help="dir utk simpan (frame, mask) utk training — "
                         "mask = geometri genangan (ground truth sempurna)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duras = args.rise_minutes * 60
    # rise di-bake dalam SATU pass sumber; out-fps mengatur lama tayang.
    frames_rise = max(n_total * 0.9, 1)
    out_fps = args.out_fps if args.out_fps > 0 else (n_total / duras if duras else fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, out_fps, (w, h))

    # noise riak bergerak
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 40, size=(h // 4, w // 4), dtype=np.uint8)

    i = 0
    mask_n = 0
    if args.dump_masks:
        os.makedirs(args.dump_masks, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = min(i / max(frames_rise, 1), 1.0)
        # kurva percepatan: lambat di awal, eksponensial (khas bandang)
        rise = args.max_rise_px * (t ** 2)
        y0 = int(h * 0.55 - rise)          # garis air simulasi
        y0 = max(0, min(y0, h - 5))
        if y0 < h - 5:
            # air keruh: pergeseran warna biru-cokelat + noise bergerak
            water = frame[y0:, :, :].astype(np.float32)
            shift = (i // 2) % (h // 4)
            ripple = cv2.resize(noise, (w, h - y0)).astype(np.float32)
            ripple = np.roll(ripple, shift, axis=0)[:, :w]
            brown = np.zeros_like(water)
            brown[..., 0] = 40; brown[..., 1] = 70; brown[..., 2] = 95
            water = water * (1 - args.turbidity) + brown * args.turbidity
            water += (ripple[..., None] - 20) * 0.9
            frame[y0:, :, :] = np.clip(water, 0, 255).astype(np.uint8)
        vw.write(frame)
        if args.dump_masks and mask_n % 10 == 0:
            stem = os.path.splitext(os.path.basename(args.out))[0]
            cv2.imwrite(os.path.join(args.dump_masks, f"{stem}_{mask_n:05d}.jpg"),
                        frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            m = np.zeros((h, w), dtype=np.uint8)
            m[y0:, :] = 255
            cv2.imwrite(os.path.join(args.dump_masks, f"{stem}_{mask_n:05d}.png"), m)
        mask_n += 1
        i += 1

    cap.release()
    vw.release()
    print(f"Selesai: {args.out} ({i} frames, rise {args.max_rise_px}px "
          f"dalam {args.rise_minutes} menit tayang)")


if __name__ == "__main__":
    main()
