from django.contrib import admin
from django.urls import path
from monitor import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", views.dashboard, name="dashboard"),
    path("api/status", views.api_status, name="api_status"),
    path("api/readings", views.api_readings, name="api_readings"),
    path("api/alerts", views.api_alerts, name="api_alerts"),
    path("stream/<str:camera_id>", views.stream, name="stream"),
    path("snapshot/<str:camera_id>", views.snapshot_latest, name="snapshot"),
]
