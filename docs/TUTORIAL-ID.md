# 📘 RiverWatch — Tutorial Lengkap (Setup → CCTV → Produksi 24 Jam)

> Sistem pemantauan sungai via CCTV: segmentasi air (YOLOv8-seg) → estimasi
> tinggi muka air → deteksi **banjir bandang** (rise-rate) → alert Telegram +
> dashboard live. Lanjutan tugas akhir *river-segmentation*.

Bagian 1: [Setup di Laptop/Kos](#1-setup-server--laptop) ·
Bagian 2: [Menambah CCTV](#2-menambah-kamera-cctv) ·
Bagian 3: [Kalibrasi](#3-kalibrasi-tinggi-muka-air) ·
Bagian 4: [Telegram](#4-alert-telegram) ·
Bagian 5: [Auto-start 24 Jam](#5-menjalankan-24-jam) ·
Bagian 6: [Troubleshooting](#6-troubleshooting)

---

## 0. Arsitektur 30 Detik

```
CCTV (RTSP/webcam/file)
   └─ CameraWorker  (thread per kamera, auto-reconnect, ambil frame TERBARU)
        └─ Detektor YOLOv8-seg  (mask "air" per frame)
             └─ LevelTracker    (garis air → cm, baseline "normal" auto)
                  └─ RiskAssessor  (NORMAL → WASPADA → SIAGA → BAHAYA)
                       ├─ Dashboard web + stream MJPEG   (proses `web`)
                       ├─ Database SQLite (histori)      (proses `monitor`)
                       └─ Alert Telegram (foto + level)  (proses `monitor`)
```

Dua proses terpisah (`monitor` dan `web`) — kalau web mati, monitoring &
alert tetap jalan. Keduanya baca/tulis lewat file status + SQLite.

---

## 1. Setup Server / Laptop

### 1.1 Kebutuhan

| Barang | Minimum | Catatan |
|---|---|---|
| OS | Linux (Ubuntu 22.04+) atau Windows 10/11 | GPU NVIDIA opsional (CPU jalan, lebih lambat) |
| Python | 3.10 – 3.12 | cek `python3 --version` |
| RAM | 4 GB (heuristik) / 8 GB (YOLO) | |
| Disk | ± 8 GB | torch + dataset |
| GPU | CUDA 12.x untuk percepatan | tanpa GPU pakai `--cpu` (fps turun) |

### 1.2 Linux — sekali jalan

```bash
git clone https://github.com/Ihsanul-Ikhwan/river-segmentation.git
cd river-segmentation
git checkout v2
./riverwatch.sh setup          # venv + deps + migrasi DB
./riverwatch.sh start          # monitor + dashboard
./riverwatch.sh status         # semua kamera harus [OK]
# buka http://localhost:8095
```

### 1.3 Windows (PowerShell)

```powershell
git clone https://github.com/Ihsanul-Ikhwan/river-segmentation.git
cd river-segmentation; git checkout v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# Windows: torch CUDA (kalau ada GPU NVIDIA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
cd webapp
python manage.py migrate
# JALAN 1 (monitor):
python manage.py monitor
# JALAN 2 (dashboard, buka terminal baru):
python manage.py runserver 0.0.0.0:8095 --noreload
```

### 1.4 Model YOLO

Weight `data/weights/water_yolov8n.pt` **tidak ikut di git** (terlalu besar
untuk repo). Tiga cara mendapatkannya:

1. **Salin** file `water_yolov8n.pt` dari mesin yang sudah training ke
   `data/weights/`.
2. **Training sendiri** (perlu GPU, ± 10 menit di RTX/Spark):
   ```bash
   .venv/bin/python scripts/build_dataset.py collect   # kumpulin gambar air
   .venv/bin/python scripts/build_dataset.py pseudo    # auto-label SegFormer
   .venv/bin/python scripts/build_dataset.py export    # format YOLO
   .venv/bin/python scripts/train_water.py --epochs 100
   ```
3. **Tanpa weight**: sistem otomatis fallback ke detektor heuristik
   (HSV warna air + riak) — cukup buat lihat alurnya, kurang akurat.

### 1.5 Uji cepat (tanpa CCTV)

Bawaan repo sudah ada 3 kamera demo (2 klip video sungai + 1 **simulasi banjir
bandang**). Simulasi menaikkan air 280px dalam 3 menit — amati dashboard
menuju `WASPADA → SIAGA → BAHAYA`. Ini bukti seluruh rantai deteksi bekerja.

---

## 2. Menambah Kamera CCTV

Semua kamera didefinisikan di **`config/cameras.yaml`** satu file.
Setelah mengubah: `./riverwatch.sh stop && ./riverwatch.sh start`.

### 2.1 IP Camera RTSP (paling umum — Hikvision / Dahua / TP-Link / Ezviz)

Format URL RTSP per merek (ganti `user:pass@ip`):

| Merek | URL |
|---|---|
| Hikvision | `rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@192.168.1.64:554/cam/realmonitor?channel=1&subtype=0` |
| TP-Link (Tapo) | `rtsp://user:pass@192.168.1.64:554/stream1` |
| Ezviz | `rtsp://user:pass@192.168.1.64:554/h264/ch1/main/av_stream` |
| Umum/ONVIF | cek manual `rtsp://ip:554/onvif1` atau dokumentasi |

> **Coba dulu di VLC/ffprobe** sebelum masuk config:
> `ffprobe "rtsp://user:pass@ip:554/..."` — keluar durasi/stream info = URL benar.
> Aktifkan dulu RTSP/ONVIF di web admin kamera (sering nonaktif default).

```yaml
cameras:
  - id: ciliwung-01                  # unik, tanpa spasi
    name: "CCTV Sungai Ciliwung"
    source: rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101
    fps: 5                           # batas analisis (hemat GPU) — 3..8 cukup
    detector: yolo
    detector_opts:
      weight_path: data/weights/water_yolov8n.pt
    calibration:
      datum_y: 720                   # lihat Bagian 3
      cm_per_px: 1.35
    risk:
      level_cm: [80, 180, 300]       # ambang WASPADA/SIAGA/BAHAYA
      rise_cm_per_min: [8, 20, 40]
```

Sistem sudah **auto-reconnect**: kamera mati / WiFi putus / ISP gangguan →
worker retry terus, badge jadi `SUSUT`, dan sambung lagi sendiri saat hidup.

### 2.2 Webcam USB

```yaml
  - id: webcam-kosan
    name: "Webcam Jendela Kos"
    source: 0            # angka = /dev/video0 (Windows: 0 juga)
    fps: 5
    detector: yolo
    detector_opts:
      weight_path: data/weights/water_yolov8n.pt
```

### 2.3 Kamera HP (Android → RTSP)

Aplikasi **IP Webcam** (Android): set mode stream = RTSP, port 554,
lalu pakai `rtsp://user:pass@ip-hp:8554/1` — atau mode MJPEG HTTP
`http://ip-hp:8080/video` (langsung didukung karena sumber http).

### 2.4 File video (demo / rekaman)

```yaml
  - id: rekaman-banjir
    name: "Rekaman 2024"
    source: data/samples/flood_sim.mp4   # relatif ke repo
    loop: true
```

### 2.5 Aturan penting

- `id` harus **unik** (sistem menolak ID ganda saat start).
- `fps` bukan fps kamera — cuma batas analisis. 3–5 cukup utk banjir
  (perubahan air lambat kecuali bandang), hemat GPU untuk banyak kamera.
- `notify: false` → catat alert ke DB tanpa kirim Telegram (pakai utk demo).
- Satu kamera error **tidak menjatuhkan** kamera lain.

---

## 3. Kalibrasi Tinggi Muka Air

Dua angka saja. Pakai **screenshot frame** dari
`http://<ip>:8095/frame/<id>` (atau lihat stream).

1. **`datum_y`** — nomor baris pixel (0 = paling atas) yang jadi acuan "0 cm".
   Idealnya garis dasar patok meteran yang terlihat di air, atau dasar sungai.
   Contoh: frame 1080 px tinggi, dasar patok muncul di baris 720 → `datum_y: 720`.
   (Screenshot frame: `http://<ip>:8095/snapshot/<id>` atau dari stream.)
2. **`cm_per_px`** — skala vertikal. Cara paling akurat:
   - Taruh patok meteran nyata di pinggir sungai (terlihat kamera).
   - Di screenshot, ukur jarak pixel antara tanda 0 cm dan 100 cm → misal 74 px.
   - `cm_per_px = 100 / 74 = 1.35`.
3. **(Opsional) `roi_polygon`** — korangan area sungai saja, buang jalan/rooftop
   agar refleksi & gerakan di luar sungai tak mengganggu:
   ```yaml
   roi_polygon: [[200,300],[1080,280],[1280,720],[0,720]]
   ```

### Mode relatif vs absolut

- Default (`relative_level: true`): sistem **belajar sendiri** level "normal"
  dari 45 detik pertama, lalu status menilai *naik berapa cm dari normal*.
  Kamera tanpa patok pun tetap berguna. ⚠️ Jangan nyalakan tepat saat banjir
  sedang naik — baseline akan ikut salah (restart setelah air normal).
- Kamera berpatok akurat: set `relative_level: false` → ambang judged absolut.

### Saran ambang `risk`

Banjir **bandang** = naik cepat. Acuan kasar (sesuaikan karakteristik sungai):

| Tingkat | level_cm (relatif) | rise_cm_per_min |
|---|---|---|
| WASPADA | +80 | +8 |
| SIAGA | +180 | +20 |
| BAHAYA | +300 | +40 |

Sistem ambil sinyal yang **lebih tinggi** dari dua kolom; naik tingkat
langsung, turun tingkat butuh 60 detik konsisten + hysteresis 10 cm (anti
status naik-turun tiap riak).

---

## 4. Alert Telegram

1. Buat bot: chat [@BotFather](https://t.me/BotFather) → `/newbot` → dapat token.
2. Kirim pesan apa pun ke botmu, lalu buka
   `https://api.telegram.org/bot<TOKEN>/getUpdates` → ambil `chat.id`.
3. Simpan di `~/.hermes/.env` (atau env var mesin):
   ```bash
   TELEGRAM_BOT_TOKEN=***
   RIVERWATCH_CHAT_ID=123456789
   ```
4. Restart monitor. Uji: nyalakan `sim-bandang` / tunggu perubahan stage →
   HP berbunyi: **foto kondisi + level + trend**, cooldown 3 menit per
   kamera+stage (anti spam).

---

## 5. Menjalankan 24 Jam

### Linux — systemd (disarankan, auto-restart kalau crash)

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/riverwatch.service <<'EOF'
[Unit]
Description=RiverWatch monitor + web
After=network-online.target

[Service]
WorkingDirectory=%h/river-segmentation/webapp
ExecStart=%h/river-segmentation/.venv/bin/python manage.py monitor
Restart=always
RestartSec=5

[Install]
WantedBy=default.target
EOF
# (buat juga riverwatch-web.service dgn ExecStart=...runserver 0.0.0.0:8095 --noreload)
systemctl --user daemon-reload
systemctl --user enable --now riverwatch riverwatch-web
loginctl enable-linger $USER        # tetap hidup walau logout
```

### Tanpa systemd

`./riverwatch.sh start` (nohup) + daftarkan di autostart desktop / `crontab @reboot`:
```
@reboot cd ~/river-segmentation && ./riverwatch.sh start
```

### Windows

`taskschd.msc` → Create Task → "At log on" →
`cmd /c cd /d C:\river-segmentation && start .\run_monitor.bat && start .\run_web.bat`
(atau pakai [NSSM](https://nssm.cc) kalau mau jadi service sungguhan).

### Pemeliharaan otomatis
- **Pruning**: reading/alert >30 hari + file snapshot lama dibuang otomatis
  tiap hari (db tidak membengkak).
- Log: `journalctl --user -u riverwatch -f` atau `tail -F data/run/*.log`.

---

## 6. Troubleshooting

| Gejala | Penyebab mungkin | Solusi |
|---|---|---|
| Kamera `OFFLINE`, error "Tidak bisa buka" | URL/file salah, kamera tak menyala | cek dengan VLC/ffprobe; cek path relatif (relatif ke **repo**, bukan cwd) |
| RTSP cuma 1-2 fps / patah-patah | UDP diblok switch/NAT | sudah otomatis dipaksa TCP; coba `subtype=1` (stream sub) utk bandwidth |
| RTSP `DESCRIBE failed: 404` | path belum di-publish / salah path | cek URL; pada mediamtx: publish dulu |
| RTSP minta password terus | karakter spesial di password | URL-encode: `@`→`%40`, `#`→`%23` |
| Layar dashboard hitam | monitor belum start / kamera belum keluar frame | `./riverwatch.sh status`; lihat error di status |
| Status `SUSUT` | frame berhenti >15 detik | cek kamera/jaringan; sistem reconnect sendiri |
| Semua BAHAYA padahal normal | kamera start saat air tinggi (baseline keracunan) atau heuristik salah (langit dikira air) | restart saat air normal (baseline belajar ulang); pakai detector `yolo` + weight benar |
| Tidak ada pesan Telegram | token/chat_id salah; `notify: false` | cek `~/.hermes/.env`; lihat kolom `notified` di tabel alert |
| `cuda out of memory` | GPU penuh model lain | turunkan `fps`/matikan model lain; atau `device: cpu` di detector_opts |
| Web tidak bisa dibuka dari HP | port diblok firewall | `sudo ufw allow 8095` ; Windows: izinkan di Defender Firewall |
| `.venv` error setelah pindah mesin | venv tidak portabel | hapus folder `.venv` lalu `./riverwatch.sh setup` ulang |
| Chmod `Permission denied` riverwatch.sh | belum executable | `chmod +x riverwatch.sh` |

**Debug berjenjang** dari paling murah:
1. `./riverwatch.sh status` → siapa mati?
2. `tail -50 data/run/monitor.log` (atau `journalctl --user -u riverwatch`)
3. `curl -s localhost:8095/api/status | python3 -m json.tool` → error per kamera
4. `cat data/frames/<id>.status.json` → status mentah dari pipeline
5. `.venv/bin/python -m pytest tests/ -q` → regresi inti harus 17 hijau

---

## 7. Ringkasan Perintah

```bash
./riverwatch.sh setup|start|stop|status|logs|demo   # launcher
.venv/bin/python -m pytest tests/ -q               # 17 test regresi
scripts/train_water.py --help                      # retrain model
scripts/make_flood_sim.py --help                   # bikin video simulasi uji
curl localhost:8095/api/status                     # API JSON
```

API: `/api/status` · `/api/readings?camera=&hours=` · `/api/alerts` ·
`/stream/<id>` (MJPEG) · `/snapshot/<id>` (JPEG terbaru) · `/admin` (Django).

Akses `/admin` (buat akun sekali): `python manage.py createsuperuser`.
Ganti `RIVERWATCH_SECRET_KEY` via env sebelum dipakai di jaringan terbuka.
