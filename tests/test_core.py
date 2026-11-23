"""RiverWatch — test regresi inti (non-hardware).

Jalankan: .venv/bin/python -m pytest tests/ -q
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.level import Calibration, LevelTracker          # noqa: E402
from pipeline.risk import (BAHAYA, NORMAL, SIAGA, WASPADA,     # noqa: E402
                           RiskAssessor, RiskConfig)
from pipeline.sources import FileSource, SourceError, build_source  # noqa: E402


# ---------------------------------------------------------------- risk
def test_stage_naik_langsung_dan_turun_perlu_konfirmasi():
    r = RiskAssessor(RiskConfig(level_cm=[10, 20, 30], rise_cm_per_min=[5, 15, 30],
                                down_confirm_sec=10))
    t = 1000.0
    stage, ch = r.assess(25, None, ts=t)          # level 25 ≥ 20 → SIAGA
    assert (stage, ch) == (SIAGA, True)
    stage, ch = r.assess(0, None, ts=t + 1)       # langsung turun? TAHAN dulu
    assert (stage, ch) == (SIAGA, False)
    stage, ch = r.assess(0, None, ts=t + 12)      # konsisten <10 selama >10 dtk
    assert (stage, ch) == (NORMAL, True)


def test_hysteresis_menahan_flap_di_ambang():
    r = RiskAssessor(RiskConfig(level_cm=[10, 20, 30], hysteresis_cm=3,
                                down_confirm_sec=0))
    r.assess(15, None, ts=1.0)                     # WASPADA
    stage, _ = r.assess(8.5, None, ts=2.0)         # <10 tapi ≥ 10-3 → TAHAN
    assert stage == WASPADA
    stage, _ = r.assess(5, None, ts=3.0)           # < ambang-hysteresis → turun
    assert stage == NORMAL


def test_rise_rate_memicu_bahaya_meski_level_rendah():
    r = RiskAssessor(RiskConfig(level_cm=[10, 20, 30], rise_cm_per_min=[5, 15, 30]))
    stage, _ = r.assess(2, 40, ts=1.0)             # level 2 cm tapi +40 cm/menit
    assert stage == BAHAYA


def test_rate_negatif_surut_diuabaikan():
    r = RiskAssessor(RiskConfig())
    stage, _ = r.assess(None, -500, ts=1.0)
    assert stage == NORMAL


def test_normal_ke_normal_tidak_dianggap_perubahan():
    r = RiskAssessor(RiskConfig())
    r.assess(None, None, ts=1.0)
    stage, ch = r.assess(None, None, ts=2.0)
    assert (stage, ch) == (NORMAL, False)


# ---------------------------------------------------------------- level
def _mask_waterline(h=720, w=1280, y=400, x0=100, x1=1180):
    """Masker sintetis: air di bawah baris y, rentang x x0..x1."""
    m = np.zeros((h, w), dtype=bool)
    m[y:, x0:x1] = True
    return m


def test_waterline_terdeteksi_dan_kalibrasi_cm():
    calib = Calibration(datum_y=700, cm_per_px=2.0)
    tr = LevelTracker(calib, baseline_window=0.1)
    s = tr.update(_mask_waterline(y=400), ts=100.0)
    assert s.waterline_y == pytest.approx(400, abs=1)
    assert s.level_cm == pytest.approx((700 - 400) * 2.0, rel=0.01)


def test_mask_kecil_ditolak():
    tr = LevelTracker(Calibration(min_water_px=500), baseline_window=0.1)
    m = np.zeros((720, 1280), dtype=bool)
    m[600:, 10:30] = True                          # 20*120=2400 px tapi sempit
    # rasio per baris = 20/1280 << 0.30 → tidak ada baris lolos
    s = tr.update(m, ts=1.0)
    assert s.waterline_y is None


def test_rise_rate_positif_saat_air_naik():
    tr = LevelTracker(Calibration(datum_y=700, cm_per_px=1.0),
                      baseline_window=0.0)
    t = 1000.0
    y = 300
    for i in range(30):                             # naik 3px tiap 0.5 dtk
        tr.update(_mask_waterline(y=y), ts=t + i * 0.5)
        y -= 3
    rate = tr.rise_rate_cm_per_min(lookback_sec=15)
    assert rate is not None and rate > 300          # 3px/0.5dtk = 6cm/0.5s = 360


def test_baseline_dipelajari_dari_window_pertama():
    tr = LevelTracker(Calibration(datum_y=700, cm_per_px=1.0), baseline_window=5.0)
    for i in range(10):
        tr.update(_mask_waterline(y=400 + (i % 3)), ts=100.0 + i * 0.4)
    assert tr.baseline_cm == pytest.approx(300, abs=2)
    for i in range(8):  # median filter span-5 butuh beberapa frame utk bergeser
        s = tr.update(_mask_waterline(y=250), ts=200.0 + i * 0.2)  # +150px
    assert s.above_normal_cm == pytest.approx(150, abs=3)


def test_roi_polygon_membatasi_area():
    calib = Calibration(datum_y=700, cm_per_px=1.0,
                        roi_polygon=[[0, 0], [640, 0], [640, 720], [0, 720]])
    tr = LevelTracker(calib, baseline_window=0.1)
    s = tr.update(_mask_waterline(y=400), ts=1.0)
    # air x100..1180; ROI x<640 → bagian dlm ROI 100..640 = 540/640 > 0.3
    assert s.waterline_y == pytest.approx(400, abs=1)
    # ROI kanan (x>700): air cuma sampai x700 → kosong → None
    calib2 = Calibration(datum_y=700, roi_polygon=[[700, 0], [1280, 0],
                                                   [1280, 720], [700, 720]])
    tr2 = LevelTracker(calib2, baseline_window=0.1)
    assert tr2.update(_mask_waterline(y=400, x0=100, x1=700),
                      ts=1.0).waterline_y is None


# ---------------------------------------------------------------- sources
def test_build_source_file_hilang_melempar_source_error(tmp_path):
    with pytest.raises(SourceError):
        build_source(str(tmp_path / "tidak-ada.mp4"))


def test_filesource_pacing_realtime(tmp_path):
    """10 frame dari video 30fps harus makan ~10/30 detik, bukan ~0."""
    import subprocess

    vid = tmp_path / "smp4.mp4"
    subprocess.run(
        [os.environ.get("FFMPEG", "ffmpeg"), "-y", "-f", "lavfi",
         "-i", "testsrc=size=128x72:rate=30:duration=2",
         "-pix_fmt", "yuv420p", str(vid)],
        capture_output=True, check=True)
    src = build_source(str(vid))
    assert isinstance(src, FileSource)
    t0 = time.monotonic()
    n = 0
    while n < 10 and src.read() is not None:
        n += 1
    elapsed = time.monotonic() - t0
    assert n == 10
    assert 0.2 < elapsed < 0.8                       # ~0.33 dtk utk 10@30fps


def test_stream_source_url_mati_tidak_menggantung():
    from pipeline.sources import StreamSource
    with pytest.raises(SourceError):
        StreamSource("rtsp://127.0.0.1:1/nope")      # port ditutup → error cepat
