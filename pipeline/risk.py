"""Logika risiko banjir bandang: threshold bertingkat + rise-rate + hysteresis.

Empat tingkat: NORMAL(0) < WASPADA(1) < SIAGA(2) < BAHAYA(3).

Kenaikan tingkat dipicu oleh DUA sinyal (ambil yang lebih tinggi):
  - level_cm  terhadap ambang [waspada, siaga, bahaya]
  - rise_rate cm/menit terhadap ambang [waspada, siaga, bahaya]
    (banjir bandang: level belum tinggi tapi naik 15 cm/10 menit = BAHAYA)

Hysteresis: turun tingkat butuh nilai < (ambang - margin) supaya status
tidak flapping di batas. Naik tingkat langsung, turun tingkat setelah
`down_confirm_sec` konsisten di bawah.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

NORMAL, WASPADA, SIAGA, BAHAYA = 0, 1, 2, 3
STAGE_NAME = {NORMAL: "NORMAL", WASPADA: "WASPADA", SIAGA: "SIAGA", BAHAYA: "BAHAYA"}
STAGE_EMOJI = {NORMAL: "🟢", WASPADA: "🟡", SIAGA: "🟠", BAHAYA: "🔴"}


@dataclass
class RiskConfig:
    level_cm: list = field(default_factory=lambda: [50, 150, 250])
    rise_cm_per_min: list = field(default_factory=lambda: [5, 15, 30])
    hysteresis_cm: float = 10.0
    hysteresis_rate: float = 2.0
    down_confirm_sec: float = 60.0

    @classmethod
    def from_dict(cls, d: dict | None) -> "RiskConfig":
        d = d or {}
        return cls(
            level_cm=list(d.get("level_cm", [50, 150, 250])),
            rise_cm_per_min=list(d.get("rise_cm_per_min", [5, 15, 30])),
            hysteresis_cm=float(d.get("hysteresis_cm", 10.0)),
            hysteresis_rate=float(d.get("hysteresis_rate", 2.0)),
            down_confirm_sec=float(d.get("down_confirm_sec", 60.0)),
        )


def _stage_from(value: float | None, thresholds: list, hyst: float,
                current: int) -> int:
    """Stage dari satu sinyal. Kalau di zona hysteresis → pertahankan current."""
    if value is None:
        return 0
    stage = 0
    for i, thr in enumerate(thresholds[:3]):
        if value >= thr:
            stage = i + 1
    # cek apakah current stage masih "aman" (di bawah ambang - hysteresis?)
    if current > stage:
        thr_current = thresholds[current - 1]
        if value >= thr_current - hyst:
            return current  # masih di zona hysteresis → tahan
    return stage


@dataclass
class RiskAssessor:
    config: RiskConfig
    current: int = NORMAL
    _below_since: float | None = None

    def __post_init__(self):
        self._below_since = None

    def assess(self, level_cm: float | None,
               rise_cm_per_min: float | None,
               ts: float | None = None) -> tuple[int, bool]:
        """Kembalikan (stage, berubah_naik_atau_turun)."""
        ts = ts or time.time()
        s_level = _stage_from(level_cm, self.config.level_cm,
                              self.config.hysteresis_cm, self.current)
        # rise-rate hanya hitung kenaikan (rate negatif = surut → 0)
        rate = rise_cm_per_min if (rise_cm_per_min or 0) > 0 else None
        s_rate = _stage_from(rate, self.config.rise_cm_per_min,
                             self.config.hysteresis_rate, self.current)
        raw = max(s_level, s_rate)

        if raw > self.current:
            self.current = raw
            self._below_since = None
            return self.current, True

        if raw < self.current:
            if self._below_since is None:
                self._below_since = ts
            changed = False
            if ts - self._below_since >= self.config.down_confirm_sec:
                self.current = raw
                self._below_since = None
                changed = True
            return self.current, changed

        self._below_since = None
        return self.current, False
