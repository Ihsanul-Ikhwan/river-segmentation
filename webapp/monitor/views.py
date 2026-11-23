"""Views RiverWatch: dashboard, JSON API, MJPEG stream per kamera."""
from __future__ import annotations

import json
import os
import time

from django.conf import settings
from django.db.models import Avg, Max, Min
from django.http import HttpResponse, JsonResponse, StreamingHttpResponse
from django.shortcuts import render

from .models import Alert, Reading

FRAMES = settings.RIVERWATCH["FRAMES_DIR"]


def _cams() -> list[dict]:
    """Daftar kamera dari yaml konfigurasi (sumber kebenaran): id + nama."""
    import yaml

    path = settings.RIVERWATCH["CAMERAS_YAML"]
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        out = []
        for c in data.get("cameras", []):
            cid = str(c["id"])
            out.append({"id": cid, "name": c.get("name", cid)})
        return out
    except OSError:
        return []


def _cam_ids() -> list[str]:
    return [c["id"] for c in _cams()]


def _read_status(camera_id: str) -> dict:
    fp = os.path.join(FRAMES, f"{camera_id}.status.json")
    try:
        with open(fp, encoding="utf-8") as f:
            st = json.load(f)
        st["fresh"] = time.time() - st.get("updated", 0) < 15
        return st
    except (OSError, json.JSONDecodeError):
        return {"stage": "OFFLINE", "stage_code": -1, "fresh": False}


def dashboard(request):
    cams = []
    latest_alert = Alert.objects.first()
    for spec in _cams():
        cid = spec["id"]
        st = _read_status(cid)
        st.setdefault("name", spec["name"])
        last = Reading.objects.filter(camera_id=cid).first()
        cams.append({
            "id": cid,
            "name": spec["name"],
            "status": st,
            "last_reading": last,
            "stream_url": f"/stream/{cid}",
        })
    recent_alerts = Alert.objects.all()[:12]
    return render(request, "dashboard.html", {
        "cameras": cams,
        "recent_alerts": recent_alerts,
        "latest_alert": latest_alert,
    })


def api_status(request):
    out = []
    for cid in _cam_ids():
        out.append({"camera_id": cid, **_read_status(cid)})
    return JsonResponse({"cameras": out, "ts": time.time()})


def api_readings(request):
    cid = request.GET.get("camera", "")
    try:
        hours = min(max(float(request.GET.get("hours", 24)), 0.1), 24 * 30)
    except ValueError:
        hours = 24.0
    from django.utils import timezone
    since = timezone.now() - timezone.timedelta(hours=hours)
    qs = Reading.objects.filter(ts__gte=since)
    if cid:
        qs = qs.filter(camera_id=cid)
    data = list(qs.order_by("ts").values("ts", "camera_id", "level_cm",
                                         "water_ratio", "rise_rate_cpm", "stage")
                 [:20000])   # rem payload — jangan serahkan jutaan titik
    for d in data:
        d["ts"] = d["ts"].isoformat()
    return JsonResponse({"readings": data, "count": len(data)})


def api_alerts(request):
    data = list(Alert.objects.values("ts", "camera_id", "stage", "message",
                                     "level_cm", "rise_rate_cpm", "notified")[:50])
    for d in data:
        d["ts"] = d["ts"].isoformat()
    return JsonResponse({"alerts": data})


def _mjpeg_generator(camera_id: str, interval: float = 0.2,
                     stale_timeout: float = 30.0):
    """Sajikan data/frames/<id>.jpg sebagai multipart MJPEG (poll 5 fps)."""
    fp = os.path.join(FRAMES, f"{camera_id}.jpg")
    boundary = "riverwatchframe"
    last_mtime = 0.0
    while True:
        try:
            mtime = os.path.getmtime(fp)
            if mtime != last_mtime:
                with open(fp, "rb") as f:
                    jpg = f.read()
                last_mtime = mtime
                yield (f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                       f"Content-Length: {len(jpg)}\r\n\r\n").encode()
                yield jpg + b"\r\n"
            else:
                time.sleep(interval)
        except OSError:
            yield (f"--{boundary}\r\nContent-Type: text/plain\r\n\r\n"
                   f"no frame yet\r\n").encode()
            time.sleep(1.0)


def stream(request, camera_id: str):
    if camera_id not in _cam_ids():
        return HttpResponse("kamera tidak dikenal", status=404)
    resp = StreamingHttpResponse(
        _mjpeg_generator(camera_id),
        content_type="multipart/x-mixed-replace;boundary=riverwatchframe")
    resp["Cache-Control"] = "no-store"
    return resp


def snapshot_latest(request, camera_id: str):
    fp = os.path.join(FRAMES, f"{camera_id}.jpg")
    if not os.path.exists(fp):
        return HttpResponse("belum ada frame", status=404)
    with open(fp, "rb") as f:
        return HttpResponse(f.read(), content_type="image/jpeg")
