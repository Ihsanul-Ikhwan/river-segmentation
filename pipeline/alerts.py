"""Notifikasi. Telegram via Bot API — token dibaca dari environment.

Bot token Hermes (~/​.hermes/.env) dipakai ulang supaya alert masuk ke HP
user tanpa bikin bot baru. Tidak ada token → alert hanya log, tidak error.
"""
from __future__ import annotations

import logging
import os
import time

import requests

log = logging.getLogger("riverwatch.alerts")


def _load_token() -> tuple[str | None, str | None]:
    """Ambil (bot_token, chat_id) dari env; fallback baca ~/.hermes/.env."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("RIVERWATCH_CHAT_ID", "8757521778")  # pemilik RiverWatch
    if token and chat:
        return token, chat
    env_file = os.path.expanduser("~/.hermes/.env")
    if os.path.exists(env_file):
        try:
            with open(env_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("TELEGRAM_BOT_TOKEN=") and not token:
                        token = line.split("=", 1)[1].strip().strip('"').strip("'")
        except OSError:
            pass
    return token, chat


class TelegramNotifier:
    def __init__(self, chat_id: str | None = None, cooldown_sec: float = 120.0):
        self.token, env_chat = _load_token()
        self.chat_id = chat_id or env_chat
        self.cooldown_sec = cooldown_sec
        self._last_sent: dict[str, float] = {}
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            log.warning("Telegram alert NONAKTIF — TELEGRAM_BOT_TOKEN/"
                        "RIVERWATCH_CHAT_ID belum diset (alert hanya ke log).")

    def _throttled(self, key: str) -> bool:
        now = time.time()
        if now - self._last_sent.get(key, -1e9) < self.cooldown_sec:
            return True
        self._last_sent[key] = now
        return False

    def send_text(self, text: str, key: str | None = None) -> bool:
        if not self.enabled:
            log.info("[ALERT-LOG] %s", text)
            return False
        if key and self._throttled(key):
            return False
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text,
                      "parse_mode": "HTML", "disable_web_page_preview": True},
                timeout=10,
            )
            ok = bool(r.json().get("ok"))
            if not ok:
                log.error("Telegram sendMessage gagal: %s", r.text[:200])
            return ok
        except requests.RequestException as exc:
            log.error("Telegram error: %s", exc)
            return False

    def send_photo(self, image_bytes: bytes, caption: str,
                   key: str | None = None) -> bool:
        if not self.enabled:
            log.info("[ALERT-LOG] (foto) %s", caption)
            return False
        if key and self._throttled(key):
            return False
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendPhoto",
                data={"chat_id": self.chat_id, "caption": caption},
                files={"photo": ("alert.jpg", image_bytes, "image/jpeg")},
                timeout=15,
            )
            ok = bool(r.json().get("ok"))
            if not ok:
                log.error("Telegram sendPhoto gagal: %s", r.text[:200])
            return ok
        except requests.RequestException as exc:
            log.error("Telegram error: %s", exc)
            return False
