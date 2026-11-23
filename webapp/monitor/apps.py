from django.apps import AppConfig


class MonitorConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "monitor"
    verbose_name = "RiverWatch Monitor"

    def ready(self):
        # SQLite WAL supaya writer (proses monitor) & reader (web) bisa
        # barengan tanpa "database is locked".
        from django.db.backends.signals import connection_created

        def _wal(sender, connection, **kwargs):
            if connection.vendor == "sqlite":
                connection.cursor().execute("PRAGMA journal_mode=WAL;")
                connection.cursor().execute("PRAGMA synchronous=NORMAL;")

        connection_created.connect(_wal)
