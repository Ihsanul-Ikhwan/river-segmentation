from django.contrib import admin
from .models import Alert, Reading


@admin.register(Reading)
class ReadingAdmin(admin.ModelAdmin):
    list_display = ("ts", "camera_id", "level_cm", "rise_rate_cpm",
                    "water_ratio", "stage", "detector")
    list_filter = ("camera_id", "stage")


@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ("ts", "camera_id", "stage", "level_cm", "rise_rate_cpm",
                    "notified")
    list_filter = ("camera_id", "stage")
