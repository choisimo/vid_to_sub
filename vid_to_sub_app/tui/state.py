from __future__ import annotations

from vid_to_sub_app.db import Database

db = Database()


def close_shared_db() -> None:
    """Close the shared TUI database connection pool."""
    db.close()
