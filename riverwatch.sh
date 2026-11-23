#!/usr/bin/env bash
# =====================================================================
# RiverWatch — launcher satu perintah (tanpa systemd, cukup bash)
#
#   ./riverwatch.sh setup     # buat venv + install deps (sekali saja)
#   ./riverwatch.sh start     # jalankan monitor + web dashboard
#   ./riverwatch.sh stop      # hentikan keduanya
#   ./riverwatch.sh status    # cek hidup/mati + status kamera
#   ./riverwatch.sh logs      # log terbaru (Ctrl+C untuk berhenti)
#   ./riverwatch.sh demo      # buka dashboard di browser
# =====================================================================
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$PWD"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"
RUN="$ROOT/data/run"
mkdir -p "$RUN" data/frames data/snapshots

port_web="${RIVERWATCH_PORT:-8095}"

say(){ printf '\033[1;36m==>\033[0m %s\n' "$*"; }
die(){ printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

need_venv(){ [ -x "$PY" ] || die "venv belum ada — jalankan: ./riverwatch.sh setup"; }

pid_of(){ [ -f "$RUN/$1.pid" ] && kill -0 "$(cat "$RUN/$1.pid")" 2>/dev/null && cat "$RUN/$1.pid"; }

start_one(){ # nama, logfile, perintah...
  local name="$1" log="$2"; shift 2
  if pid_of "$name" >/dev/null; then
    say "$name sudah hidup (pid $(pid_of "$name"))"; return
  fi
  nohup "$@" >>"$log" 2>&1 &
  echo $! > "$RUN/$name.pid"
  sleep 1
  pid_of "$name" >/dev/null || { tail -5 "$log"; die "$name gagal hidup — lihat $log"; }
  say "$name hidup (pid $(pid_of "$name"))"
}

case "${1:-}" in
  setup)
    command -v python3 >/dev/null || die "python3 tidak ditemukan"
    say "buat virtualenv .venv"
    python3 -m venv .venv
    "$PY" -m pip install --upgrade pip
    say "install dependensi (beberapa menit, ~6 GB utk torch CUDA)"
    "$PY" -m pip install -r requirements.txt
    say "migrasi database"
    ( cd webapp && ../.venv/bin/python manage.py migrate )
    say "selesai. Catatan model:"
    echo "    Taruh weight YOLO di  data/weights/water_yolov8n.pt"
    echo "    (hasil training sendiri / minta link ke pemilik proyek)."
    echo "    Tanpa weight, sistem otomatis fallback ke detektor heuristik."
    ;;
  start)
    need_venv
    start_one monitor "$RUN/monitor.log" \
      "$PY" webapp/manage.py monitor
    start_one web "$RUN/web.log" \
      "$PY" webapp/manage.py runserver "0.0.0.0:$port_web" --noreload
    say "dashboard: http://localhost:$port_web  (LAN: http://<ip-mesin>:$port_web)"
    ;;
  stop)
    for n in monitor web; do
      if pid_of "$n" >/dev/null; then
        kill "$(pid_of "$n")" 2>/dev/null || true
        say "$n dihentikan"
      else say "$n tidak jalan"; fi
    done
    ;;
  status)
    for n in monitor web; do
      if pid_of "$n" >/dev/null; then echo "  $n: HIDUP (pid $(pid_of "$n"))"
      elif systemctl --user is-active -q "rw-$n" 2>/dev/null; then
        echo "  $n: HIDUP (systemd rw-$n)"
      else echo "  $n: mati"; fi
    done
    curl -s -m 3 "http://127.0.0.1:$port_web/api/status" 2>/dev/null | \
      "$PY" -c "import json,sys
d=json.load(sys.stdin)
for c in d['cameras']:
    f='OK ' if c.get('fresh') and not c.get('stale') else '--'
    print(f\"  [{f}] {c['camera_id']:14s} {c.get('stage','?'):8s} fps={c.get('fps',0)}\")" 2>/dev/null || echo "  (web belum jawab API)"
    ;;
  logs)
    tail -n 40 -F "$RUN/monitor.log" "$RUN/web.log"
    ;;
  demo)
    xdg-open "http://localhost:$port_web" 2>/dev/null || \
      sensible-browser "http://localhost:$port_web" 2>/dev/null || \
      echo "buka manual: http://localhost:$port_web"
    ;;
  *)
    grep '^#   ' "$0" | sed 's/^#   //'
    ;;
esac
