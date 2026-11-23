import os
from pathlib import Path

# BASE_DIR = .../river-segmentation/webapp ; REPO_ROOT = .../river-segmentation
BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent

SECRET_KEY = os.environ.get("RIVERWATCH_SECRET_KEY",
                            "django-insecure-riverwatch-dev-ganti-di-produksi")
# DEBUG default ON utk dev; set RIVERWATCH_DEBUG=0 utk produksi/kos-kan
DEBUG = os.environ.get("RIVERWATCH_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "monitor",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": REPO_ROOT / "data" / "riverwatch.sqlite3",
        "OPTIONS": {"timeout": 20},
    }
}

AUTH_PASSWORD_VALIDATORS = []

LANGUAGE_CODE = "id-id"
TIME_ZONE = "Asia/Jakarta"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = REPO_ROOT / "data" / "static"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# RiverWatch
RIVERWATCH = {
    "CAMERAS_YAML": str(REPO_ROOT / "config" / "cameras.yaml"),
    "FRAMES_DIR": str(REPO_ROOT / "data" / "frames"),
    "SNAPSHOTS_DIR": str(REPO_ROOT / "data" / "snapshots"),
}
