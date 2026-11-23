#!/usr/bin/env python3
"""Soak test 15 menit: RSS monitor/web, kebugaran kamera tiap menit."""
import json
import subprocess
import time
import urllib.request


def rss(pid: str) -> int:
    try:
        with open(f"/proc/{int(pid)}/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):
        pass
    return 0


def pid_of(unit: str) -> str:
    return subprocess.run(
        ["systemctl", "--user", "show", unit, "-p", "MainPID", "--value"],
        capture_output=True, text=True).stdout.strip()


mon0 = web0 = None
rows = []
for i in range(15):
    time.sleep(60)
    m, w = pid_of("rw-monitor"), pid_of("rw-web")
    if mon0 is None:
        mon0, web0 = rss(m), rss(w)
    try:
        d = json.load(urllib.request.urlopen(
            "http://127.0.0.1:8095/api/status", timeout=5))
        states = ",".join(
            f"{c['camera_id'][:4]}={'ok' if c.get('fresh') and not c.get('stale') else 'STALE'}"
            for c in d["cameras"])
    except Exception as exc:  # noqa: BLE001
        states = f"API-ERR {exc}"
    rows.append(f"t+{60*(i+1):3d}s mon={rss(m)}MB(Δ{rss(m)-mon0:+d}) "
                f"web={rss(w)}MB | {states}")
    print(rows[-1], flush=True)

bad = [r for r in rows if "STALE" in r or "API-ERR" in r]
print(f"\nSOAK_RESULT: {len(rows)} sampel, {len(bad)} masalah", flush=True)
