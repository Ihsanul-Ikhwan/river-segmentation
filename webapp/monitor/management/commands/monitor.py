"""`python manage.py monitor` — jalankan engine CCTV (proses daemon).

Engine (thread) hanya menghasilkan event; semua akses DB lewat SATU
writer thread di sini (sqlite + multi-thread = bahaya kalau langsung).
Alert Telegram dikirim dari writer thread yang sama.
"""
from __future__ import annotations

import os
import queue
import threading
import time

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from pipeline.alerts import TelegramNotifier
from pipeline.engine import Engine, load_cameras
from pipeline.risk import STAGE_EMOJI, STAGE_NAME

from monitor.models import Alert, Reading

STAGE_KEY = {1: "WASPADA", 2: "SIAGA", 3: "BAHAYA", 0: "NORMAL"}


class Command(BaseCommand):
    help = "Jalankan monitoring CCTV sungai (engine + DB + alert Telegram)."

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.q: queue.Queue = queue.Queue()
        self.stop = threading.Event()

    def add_arguments(self, parser):
        parser.add_argument("--config", default=None,
                            help="Path cameras.yaml (default dari settings)")
        parser.add_argument("--sample-interval", type=float, default=5.0,
                            help="Detik antar-penyimpanan reading ke DB")
        parser.add_argument("--alert-cooldown", type=float, default=180.0,
                            help="Detik cooldown notifikasi per kamera+stage")

    def handle(self, *args, **opts):
        yaml_path = opts["config"] or settings.RIVERWATCH["CAMERAS_YAML"]
        cams = load_cameras(yaml_path)
        self.sample_interval = opts["sample_interval"]
        self.notifier = TelegramNotifier()
        snap_dir = settings.RIVERWATCH["SNAPSHOTS_DIR"]
        os.makedirs(snap_dir, exist_ok=True)

        engine = Engine(cams, on_reading=self._on_event)
        writer = threading.Thread(target=self._writer_loop, daemon=True,
                                  name="db-writer")
        writer.start()
        engine.start()
        self.stdout.write(self.style.SUCCESS(
            f"RiverWatch monitor hidup: {len(cams)} kamera. Ctrl+C untuk berhenti."))

        try:
            while True:
                time.sleep(2)
        except KeyboardInterrupt:
            self.stdout.write("Menghentikan engine...")
            engine.stop()
            self.stop.set()
            writer.join(timeout=5)
            self.stdout.write(self.style.SUCCESS("Bersih. Sampai jumpa."))

    # ---------- engine callback (thread pipeline) ----------
    def _on_event(self, cam_cfg, stage, sample, rise, changed, jpeg):
        self.q.put({
            "cam": cam_cfg, "stage": stage, "sample": sample, "rise": rise,
            "changed": changed, "jpeg": jpeg, "t": time.time(),
        })

    # ---------- satu-satunya penulis DB ----------
    def _writer_loop(self):
        last_saved: dict[str, float] = {}
        last_stage: dict[str, int] = {}
        while not self.stop.is_set():
            try:
                ev = self.q.get(timeout=1.0)
            except queue.Empty:
                continue
            try:
                self._handle(ev, last_saved, last_stage)
            except Exception as exc:  # noqa: BLE001
                self.stderr.write(f"writer error: {exc}")

    def _handle(self, ev, last_saved, last_stage):
        cam = ev["cam"]
        cid = cam.id
        stage, sample, rise = ev["stage"], ev["sample"], ev["rise"]

        # 1) simpan reading (di-throttle per kamera)
        if time.time() - last_saved.get(cid, 0) >= self.sample_interval:
            Reading.objects.create(
                camera_id=cid,
                level_cm=None if sample.level_cm is None else round(sample.level_cm, 2),
                water_ratio=round(sample.water_ratio, 4),
                rise_rate_cpm=None if rise is None else round(rise, 3),
                stage=stage,
                detector=cam.detector,
            )
            last_saved[cid] = time.time()

        # 2) perubahan stage → Alert + snapshot + Telegram
        prev = last_stage.get(cid)
        if prev is None:
            last_stage[cid] = stage
            return
        if stage == prev:
            return
        last_stage[cid] = stage

        snapshot_path = ""
        if ev["jpeg"]:
            fname = f"{cid}_{int(ev['t'])}.jpg"
            snapshot_path = os.path.join(
                settings.RIVERWATCH["SNAPSHOTS_DIR"], fname)
            with open(snapshot_path, "wb") as f:
                f.write(ev["jpeg"])

        lv = "—" if sample.level_cm is None else f"{sample.level_cm:.0f} cm"
        if sample.above_normal_cm is not None:
            lv += f" ({sample.above_normal_cm:+.0f} vs normal)"
        rr = "—" if rise is None else f"{rise:+.1f} cm/menit"
        msg = (f"{cam.name}: status → {STAGE_NAME[stage]} "
               f"(level {lv}, trend {rr})")

        alert = Alert.objects.create(
            camera_id=cid, stage=stage, message=msg,
            level_cm=sample.level_cm, rise_rate_cpm=rise,
            snapshot=snapshot_path)

        emoji = STAGE_EMOJI[stage]
        text = (f"{emoji} <b>RIVERWATCH — {STAGE_NAME[stage]}</b>\n"
                f"📷 {cam.name} (<code>{cid}</code>)\n"
                f"Level air: <b>{lv}</b>\nTren: <b>{rr}</b>\n"
                f"_{timezone.localtime().strftime('%d %b %H:%M WIB')}_")
        sent = False
        if getattr(cam, "notify", True):
            if ev["jpeg"]:
                sent = self.notifier.send_photo(
                    ev["jpeg"], text.replace("<b>", "").replace("</b>", "")
                    .replace("<code>", "").replace("</code>", ""),
                    key=f"{cid}:{stage}" if stage != 0 else None)
            if not sent:
                sent = self.notifier.send_text(text, key=f"{cid}:{stage}")
        alert.notified = bool(sent)
        alert.save(update_fields=["notified"])
        self.stdout.write(f"[ALERT] {msg}")
