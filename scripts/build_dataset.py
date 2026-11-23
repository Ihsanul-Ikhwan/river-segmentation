#!/usr/bin/env python
"""Build dataset segmentasi air untuk YOLOv8-seg.

Tahap:
  collect  : kumpulin gambar mentah (foto Commons CC + frame video sampel)
  pseudo   : label otomatis pakai SegFormer-ADE20K (punya class river/lake/
             waterfall/sea) → mask biner 'water'
  export   : convert ke format YOLO-seg (txt polygon per gambar) + split

Strategi ini "weak-to-strong distillation": guru segmentasi general
(SegFormer) melatih murid spesialis CCTV (YOLOv8n-seg) yang kencang di edge.

Pemakaian:
  .venv/bin/python scripts/build_dataset.py collect
  .venv/bin/python scripts/build_dataset.py pseudo
  .venv/bin/python scripts/build_dataset.py export
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

random_rng = random.Random(7)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DATASET_DIR = os.path.join(REPO, "data", "datasets", "water")
RAW_DIR = os.path.join(DATASET_DIR, "raw")
MASK_DIR = os.path.join(DATASET_DIR, "masks")
MANIFEST = os.path.join(DATASET_DIR, "manifest.json")
UA = {"User-Agent": "RiverWatch/2.0 (dataset builder; edukasi)"}

# ADE20K class yang kita golongkan 'water' (nama, bukan id — aman antar-versi)
WATER_LABELS = {"river", "sea", "ocean", "lake", "pond", "waterfall",
                "pool", "water"}
EXCLUDE_LABELS = {"swimming pool"}   # domain jauh dari sungai


# ------------------------------------------------------------------ collect
def _commons_search(query: str, limit: int, max_w: int = 1600) -> list[dict]:
    out, offset = [], 0
    while len(out) < limit:
        params = {
            "action": "query", "generator": "search",
            "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": "6",
            "gsrlimit": "50", "gsroffset": str(offset),
            "prop": "imageinfo", "iiprop": "url|size|mime",
            "iiurlwidth": str(max_w), "format": "json",
        }
        url = ("https://commons.wikimedia.org/w/api.php?" +
               urllib.parse.urlencode(params))
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            break
        for p in pages.values():
            ii = (p.get("imageinfo") or [{}])[0]
            if ii.get("mime") != "image/jpeg":
                continue
            if not (0.3e6 < ii.get("size", 0) < 25e6):
                continue
            out.append({"title": p["title"],
                        "thumb": ii.get("thumburl") or ii["url"],
                        "page": p.get("descriptionurl", "")})
        if "continue" not in data:
            break
        offset = data["continue"]["gsroffset"]
    return out[:limit]


def collect_openimages(manifest: dict, target: int = 180) -> None:
    """Sumber kedua: Open Images V7 validation split (GCS publik, tanpa limit).

    Kelas yang diminati: River (/m/01jfm_). Ambil ImageID yang berlabel
    River lalu unduh dari storage.googleapis.com/openimages/validation/.
    """
    print("== Open Images (River) ==")
    req = urllib.request.Request(
        "https://storage.googleapis.com/openimages/dataset/"
        "validation-annotations-bbox.csv", headers=UA)
    ids: list[str] = []
    with urllib.request.urlopen(req, timeout=120) as r:
        header = r.readline().decode().strip().split(",")
        id_i, lab_i = header.index("ImageID"), header.index("LabelsText")
        for line in r:
            parts = line.decode().strip().split(",")
            if len(parts) > lab_i and "/m/01jfm_" in parts[lab_i]:
                ids.append(parts[id_i])
    print(f"  {len(ids)} gambar berlabel River")
    random_rng.shuffle(ids)
    got = 0
    for iid in ids:
        if got >= target:
            break
        fname = f"oi_{iid}.jpg"
        if any(v.get("file") == fname for v in manifest.values()):
            continue
        path = os.path.join(RAW_DIR, fname)
        url = (f"https://storage.googleapis.com/openimages/"
               f"validation/annotated/{iid}.jpg")
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=45) as r2, \
                    open(path, "wb") as f:
                shutil.copyfileobj(r2, f)
            if os.path.getsize(path) < 20000:
                os.remove(path)
                continue
            manifest[f"openimages:{iid}"] = {
                "file": fname, "query": "river", "source": "openimages",
                "page": f"https://openimages.org/?i={iid}"}
            got += 1
            if got % 25 == 0:
                print(f"    {got}/{target} terunduh")
        except Exception:  # noqa: BLE001
            continue
    print(f"  Open Images selesai: +{got}")


def cmd_collect(args):
    os.makedirs(RAW_DIR, exist_ok=True)
    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            manifest = json.load(f)

    # SUMBER 1 (cepat & tanpa rate-limit): Open Images River
    try:
        collect_openimages(manifest)
        with open(MANIFEST, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=1, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        print(f"  ! Open Images gagal: {exc}")

    # SUMBER 2: frame video sampel lokal (instan)
    _collect_video_frames(manifest)
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)

    # SUMBER 3 (best-effort, rate-limited): Wikimedia Commons
    if os.environ.get("SKIP_COMMONS") != "1":
        _collect_commons(manifest)

    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(f"Selesai collect: {len(manifest)} gambar di {RAW_DIR}")


def _collect_video_frames(manifest: dict) -> None:
    import cv2
    samples = os.path.join(REPO, "data", "samples")
    for vf in sorted(os.listdir(samples)):
        if not vf.endswith((".webm", ".mp4", ".mkv")):
            continue
        cap = cv2.VideoCapture(os.path.join(samples, vf))
        i, step = 0, max(int((cap.get(cv2.CAP_PROP_FRAME_COUNT) or 100) / 25), 1)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % step == 0:
                fname = f"vid_{vf}_{i:05d}.jpg"
                cv2.imwrite(os.path.join(RAW_DIR, fname), frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                manifest[f"video:{vf}#{i}"] = {
                    "file": fname, "query": "video", "source": "local-sample"}
            i += 1
        cap.release()


def _collect_commons(manifest: dict) -> None:
    queries = {
        "river flowing water": 60, "river current rapids": 40,
        "flood flooded street water": 45, "flash flood river": 35,
        "sungai": 25, "river weir dam water": 20, "whitewater river": 25,
    }
    for q, n in queries.items():
        try:
            hits = _commons_search(q, n)
        except Exception as exc:  # noqa: BLE001
            print(f"  ! search '{q}' gagal: {exc}")
            continue
        print(f"  {q}: {len(hits)} kandidat")
        for h in hits:
            key = h["title"]
            if key in manifest:
                continue
            fname = "".join(c for c in h["title"].replace(" ", "_")
                            if c not in '\\/:*?"<>|').replace("File_", "", 1)
            fname = fname.rsplit(".", 1)[0] + ".jpg"
            path = os.path.join(RAW_DIR, fname)
            try:
                for attempt in range(4):
                    req = urllib.request.Request(h["thumb"], headers=UA)
                    try:
                        with urllib.request.urlopen(req, timeout=45) as r, \
                                open(path, "wb") as f:
                            shutil.copyfileobj(r, f)
                        manifest[key] = {"file": fname, "query": q,
                                         "source": "commons", "page": h["page"]}
                        break
                    except urllib.error.HTTPError as exc:
                        if exc.code == 429:
                            wait = 20 * (attempt + 1)
                            print(f"    429 — tunggu {wait}s…")
                            time.sleep(wait)
                            continue
                        raise
            except Exception as exc:  # noqa: BLE001
                print(f"    unduh gagal {key}: {exc}")
            time.sleep(4.0 + random_rng.random() * 2)


# ------------------------------------------------------------------- pseudo
def cmd_pseudo(args):
    import cv2
    import numpy as np
    import torch
    from PIL import Image, ImageFile
    from transformers import (SegformerForSemanticSegmentation,
                              SegformerImageProcessor)

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "nvidia/segformer-b2-finetuned-ade-512-512"
    print(f"Muat {model_name} → {device}")
    feature = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device).eval()
    id2label = model.config.id2label
    water_ids = {i for i, n in id2label.items()
                 if str(n).lower() in WATER_LABELS
                 and str(n).lower() not in EXCLUDE_LABELS}
    print("Water ADE ids:", sorted(water_ids))

    os.makedirs(MASK_DIR, exist_ok=True)
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".jpg"))
    if args.limit:
        files = files[: args.limit]
    stats = {}
    stats_path = os.path.join(DATASET_DIR, "pseudo_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
    t0 = time.time()
    for k, fname in enumerate(files):
        path = os.path.join(RAW_DIR, fname)
        if os.path.exists(os.path.join(MASK_DIR, fname)):
            # resume: sudah pernah di-label
            continue
        try:
            img = np.asarray(Image.open(path).convert("RGB"))
            h, w = img.shape[:2]
            with torch.no_grad():
                inputs = feature(images=img, return_tensors="pt").to(device)
                logits = model(**inputs).logits
                logits_up = torch.nn.functional.interpolate(
                    logits, size=(h, w), mode="bilinear", align_corners=False)
                pred = logits_up.argmax(1)[0].cpu().numpy()
            mask = np.isin(pred, list(water_ids)).astype(np.uint8) * 255
            # bersihin noise kecil
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            ratio = float(mask.mean()) / 255.0
            cv2.imwrite(os.path.join(MASK_DIR, fname), mask)
            stats[fname] = ratio
        except Exception as exc:  # noqa: BLE001 — file rusak, skip
            print(f"  skip {fname}: {exc}")
            continue
        if (k + 1) % 25 == 0:
            print(f"  {k+1}/{len(files)} ({(time.time()-t0)/(k+1):.2f}s/gbr) "
                  f"rata2 water={np.mean(list(stats.values())):.2f}")
    with open(os.path.join(DATASET_DIR, "pseudo_stats.json"), "w") as f:
        json.dump(stats, f)
    print(f"Selesai pseudo: {len(stats)} gambar, rata-rata water ratio "
          f"{np.mean(list(stats.values())):.2f}")


# ------------------------------------------------------------------- export
def _mask_to_polygons(mask, min_area_frac=0.004, epsilon_frac=0.0015):
    """Mask → list polygon ter-normalisasi.

    Filter anti-langit: komponen yang centroid-nya di 25% atas frame dan
    bukan komponen terbesar dibuang — SegFormer sering keliru langit keruh
    sebagai air, sedangkan di foto/video air hampir selalu di bawah.
    """
    import cv2
    import numpy as np

    h, w = mask.shape
    n, labels, stats_cc, cents = cv2.connectedComponentsWithStats(
        mask, 8, cv2.CV_32S)
    if n <= 1:
        return []
    areas = stats_cc[1:, cv2.CC_STAT_AREA]
    biggest = int(np.argmax(areas)) + 1
    contours = []
    for lab in range(1, n):
        m_lab = (labels == lab).astype(np.uint8) * 255
        cs, _ = cv2.findContours(m_lab, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_SIMPLE)
        for c in cs:
            if cv2.contourArea(c) < min_area_frac * h * w:
                continue
            cy = cents[lab][1]
            if cy < 0.25 * h and lab != biggest:
                continue  # patch kecil di zona langit
            eps = epsilon_frac * max(h, w)
            c = cv2.approxPolyDP(c, eps, True)
            pts = c.reshape(-1, 2).astype(float)
            pts[:, 0] /= w
            pts[:, 1] /= h
            if len(pts) >= 3:
                contours.append(pts.clip(0, 1).round(6).tolist())
    return contours


def cmd_export(args):
    import cv2

    out = os.path.join(DATASET_DIR, "yolo")
    if os.path.exists(out):
        shutil.rmtree(out)
    for split in ("train", "val"):
        os.makedirs(os.path.join(out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out, "labels", split), exist_ok=True)

    stats_path = os.path.join(DATASET_DIR, "pseudo_stats.json")
    with open(stats_path) as f:
        stats = json.load(f)

    random.seed(42)
    files = list(stats.keys())
    random.shuffle(files)
    n_val = max(1, len(files) // 6)
    splits = {"val": files[:n_val], "train": files[n_val:]}

    exported = {"train": 0, "val": 0, "skip": 0}
    for split, flist in splits.items():
        for fname in flist:
            ratio = stats[fname]
            if not (0.03 <= ratio <= 0.95):
                exported["skip"] += 1
                continue
            mask = cv2.imread(os.path.join(MASK_DIR, fname), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                exported["skip"] += 1
                continue
            polys = _mask_to_polygons(mask)
            if not polys:
                exported["skip"] += 1
                continue
            shutil.copy(os.path.join(RAW_DIR, fname),
                        os.path.join(out, "images", split, fname))
            txt = os.path.join(out, "labels", split, fname.rsplit(".", 1)[0] + ".txt")
            with open(txt, "w") as f:
                for poly in polys:
                    flat = " ".join(f"{c:.6f}" for xy in poly for c in xy)
                    f.write(f"0 {flat}\n")
            exported[split] += 1

    yaml_path = os.path.join(DATASET_DIR, "water.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"""path: {out}
train: images/train
val: images/val
names:
  0: water
""")
    print(f"Export selesai: train={exported['train']} val={exported['val']} "
          f"skip={exported['skip']}\nConfig: {yaml_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["collect", "pseudo", "export"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"collect": cmd_collect, "pseudo": cmd_pseudo, "export": cmd_export}[args.stage](args)


if __name__ == "__main__":
    main()
