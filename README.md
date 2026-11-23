# 🌊 RiverWatch v2 — Pemantauan Sungai & Deteksi Banjir Bandang via CCTV

Kelanjutan skripsi **river-segmentation** (YOLOv8-seg untuk segmentasi aliran
sungai), dikembangkan jadi sistem pantau CCTV **real-time** dengan estimasi
**tinggi muka air** dan **peringatan banjir bandang bertingkat** ke Telegram.

> **Banjir bandang ≠ air tinggi.** Yang mematikan adalah *kecepatan naik*.
> Karena itu sistem ini bukan cuma ukur level, tapi **rise-rate (cm/menit)** —
> air naik 15 cm dalam 10 menit langsung触发 SIAGA walau levelnya belum tinggi.

---

## ✨ Fitur

| Fitur | Keterangan |
|---|---|
| **Multi-kamera** | File video, webcam, atau RTSP IP-camera — dari satu `cameras.yaml` |
| **Segmentasi air (YOLOv8-seg)** | Model kelas `water` hasil training (lihat di bawah) |
| **Estimasi tinggi muka air** | Garis air per frame → kalibrasi pixel→cm |
| **Deteksi banjir bandang** | Rise-rate + threshold 4 tingkat: NORMAL/WASPADA/SIAGA/BAHAYA, dengan hysteresis anti-kedip |
| **Dashboard live** | Stream MJPEG + grafik tren + status semua kamera |
| **Alert Telegram** | Naik/turun tingkat → foto + pesan ke HP |
| **REST API** | `/api/status`, `/api/readings`, `/api/alerts` |
| **Logger DB** | Semua reading & alert tersimpan (SQLite WAL) untuk audit/tren |

## 🏗️ Arsitektur

```
config/cameras.yaml
        │
        ▼
manage.py monitor  ──►  Engine (thread/kamera)
                          CameraWorker ─► baca frame (latest-only)
                          Detector     ─► mask air (YOLO / heuristik)
                          LevelTracker ─► garis air → cm → rise-rate
                          RiskAssessor ─► stage NORMAL..BAHAYA
                          │
                          ├─► data/frames/<id>.jpg   ( utk stream web )
                          ├─► queue ─► DB writer ─► Reading / Alert
                          └─► TelegramNotifier ─► HP

manage.py runserver ──► Dashboard + MJPEG stream + JSON API (baca DB + frames)
```

Engine dan web **proses terpisah** — engine yang nulis, web yang baca. SQLite
WAL bikin keduanya bisa barengan tanpa lock.

## 🚀 Mulai cepat

```bash
# 1. Setup venv (sudah ada .venv di mesin ini; kalau baru:)
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Migration
cd webapp && ../.venv/bin/python manage.py migrate

# 3. Jalankan engine monitor (terminal 1)
../.venv/bin/python manage.py monitor

# 4. Jalankan dashboard (terminal 2)
../.venv/bin/python manage.py runserver 0.0.0.0:8090
# buka http://localhost:8090
```

## 🎓 Cara model air dibuat (weight `best.pt` skripsi hilang)

Weight asli tidak ikut ter-commit & tidak ada di history git, jadi **latih
ulang** dengan strategi *weak-to-strong distillation*:

1. **Kumpul gambar** sungai/banjir (Wikimedia Commons CC + Open Images "River"
   + frame video sampel) → `scripts/build_dataset.py collect`
2. **Pseudo-label otomatis** pakai SegFormer-ADE20K (punya class river/lake/
   waterfall) → mask biner `water` → `build_dataset.py pseudo`
3. **Export** ke format YOLO-seg + split train/val → `build_dataset.py export`
4. **Training** YOLOv8n-seg di GPU GB10 → `scripts/train_water.py`

```bash
.venv/bin/python scripts/build_dataset.py collect
.venv/bin/python scripts/build_dataset.py pseudo
.venv/bin/python scripts/build_dataset.py export
.venv/bin/python scripts/train_water.py        # hasil → data/weights/water_yolov8n.pt
```

Setelah itu, di `cameras.yaml` ganti `detector: heuristic` → `detector: yolo`.

## 🎬 Demo tanpa CCTV asli

```bash
# video simulasi banjir bandang (level air naik eksponensial)
.venv/bin/python scripts/make_flood_sim.py \
    data/samples/klamath.webm data/samples/flood_sim.mp4 --rise-minutes 5
```

## 📷 Nambah kamera CCTV (RTSP)

```yaml
cameras:
  - id: cctv-01
    name: "CCTV Ciliwung"
    source: rtsp://user:pass@192.168.1.50:554/Streaming/Channels/101
    fps: 5
    detector: yolo
    detector_opts: { weight_path: data/weights/water_yolov8n.pt }
    calibration:
      datum_y: 720       # baris pixel = 0 cm
      cm_per_px: 1.35    # kalibrasi pakai patok meteran nyata
      roi_polygon: [[200,300],[1080,280],[1280,720],[0,720]]
    risk:
      level_cm: [80,180,300]        # WASPADA/SIAGA/BAHAYA
      rise_cm_per_min: [8,20,40]
```

**Kalibrasi `cm_per_px`:** potret patok/meteran di frame, hitung
`(selisih cm nyata) / (selisih pixel)` → isi. Tanpa kalibrasi, level masih
berarti relatif (tren naik/turun tetap akurat).

## ⚠️ Batasan & catatan jujur

- **Kalibrasi level mutlak butuh patok fisik** — tanpa itu angka cm cuma
  estimasi; deteksi *tren* (bagian penting banjir bandang) tetap jalan.
- **Model air hasil distilasi** dari SegFormer — akurat untuk demo & CCTV umum,
  tapi untuk produksi sebaiknya di-fine-tune dengan data kamera spesifik.
- **Heuristik (HSV+riak)** hanya fallback; sensitif cahaya & vegetasi.
- Deteksi air **bukan** pengganti sensor telemetri resmi (BMKG/Pos Air).

## 📁 Struktur

```
pipeline/          inti, bebas-framework
  sources.py camera_worker.py engine.py level.py risk.py alerts.py
  detectors/ (base, heuristic, yolo_seg)
webapp/
  config/          settings Django
  monitor/         models, views, admin, management/commands/monitor.py
  templates/dashboard.html
scripts/           build_dataset, train_water, make_flood_sim, probe_detector
config/cameras.yaml
data/              samples, weights, datasets, frames (di-ignore)
```

---
*Dikembangkan ulang dari skripsi river-segmentation di DGX Spark GB10.*
