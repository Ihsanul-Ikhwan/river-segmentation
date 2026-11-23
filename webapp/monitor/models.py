from django.db import models

STAGE_CHOICES = [(0, "NORMAL"), (1, "WASPADA"), (2, "SIAGA"), (3, "BAHAYA")]


class Reading(models.Model):
    """Satu titik waktu hasil analisis satu kamera."""

    camera_id = models.CharField(max_length=64, db_index=True)
    ts = models.DateTimeField(auto_now_add=True, db_index=True)
    level_cm = models.FloatField(null=True, blank=True)
    water_ratio = models.FloatField(null=True, blank=True)
    rise_rate_cpm = models.FloatField(null=True, blank=True,
                                      help_text="rise-rate cm/menit")
    stage = models.SmallIntegerField(choices=STAGE_CHOICES, default=0)
    fps = models.FloatField(null=True, blank=True)
    detector = models.CharField(max_length=32, blank=True)

    class Meta:
        ordering = ["-ts"]
        indexes = [models.Index(fields=["camera_id", "-ts"])]

    def __str__(self):
        return f"{self.camera_id} {self.ts:%H:%M:%S} {self.level_cm}cm {self.get_stage_display()}"


class Alert(models.Model):
    """Perubahan status risiko (naik/turun tingkat) — permanen untuk audit."""

    camera_id = models.CharField(max_length=64, db_index=True)
    ts = models.DateTimeField(auto_now_add=True)
    stage = models.SmallIntegerField(choices=STAGE_CHOICES)
    message = models.TextField(blank=True)
    level_cm = models.FloatField(null=True, blank=True)
    rise_rate_cpm = models.FloatField(null=True, blank=True)
    snapshot = models.CharField(max_length=255, blank=True,
                                help_text="path frame saat alert")
    notified = models.BooleanField(default=False,
                                   help_text="terkirim Telegram")

    class Meta:
        ordering = ["-ts"]

    def __str__(self):
        return f"ALERT {self.camera_id} {self.get_stage_display()} @ {self.ts:%y-%m-%d %H:%M}"
