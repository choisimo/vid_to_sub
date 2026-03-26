from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

from .constants import SECRET_ENV_KEYS

KEYRING_SERVICE_NAME = "vid_to_sub"
SECRET_STORAGE_MODE_ENV = "VID_TO_SUB_SECRET_STORAGE"


def _resolve_keyring_client() -> tuple[Any | None, str | None]:
    try:
        import keyring
    except Exception as exc:
        return None, str(exc)
    return keyring, None


def secret_storage_mode() -> str:
    raw_mode = os.environ.get(SECRET_STORAGE_MODE_ENV, "auto").strip().lower()
    if raw_mode == "env":
        return "env"
    client, _error = _resolve_keyring_client()
    if raw_mode == "keyring":
        return "keyring" if client is not None else "env"
    return "keyring" if client is not None else "env"


def read_secret_value(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if value or key not in SECRET_ENV_KEYS:
        return value
    if secret_storage_mode() != "keyring":
        return ""
    client, _error = _resolve_keyring_client()
    if client is None:
        return ""
    try:
        stored = client.get_password(KEYRING_SERVICE_NAME, key)
    except Exception:
        return ""
    return str(stored or "").strip()


def hydrate_secret_env(
    keys: Iterable[str] = SECRET_ENV_KEYS,
    *,
    override: bool = False,
) -> dict[str, str]:
    if secret_storage_mode() != "keyring":
        return {}
    client, _error = _resolve_keyring_client()
    if client is None:
        return {}

    hydrated: dict[str, str] = {}
    for key in keys:
        if not override and os.environ.get(key, "").strip():
            continue
        try:
            stored = client.get_password(KEYRING_SERVICE_NAME, key)
        except Exception:
            continue
        value = str(stored or "").strip()
        if not value:
            continue
        os.environ[key] = value
        hydrated[key] = value
    return hydrated


def persist_secret_value(key: str, value: str) -> str:
    cleaned = str(value or "").strip()
    if cleaned:
        os.environ[key] = cleaned
    else:
        os.environ.pop(key, None)

    if key not in SECRET_ENV_KEYS or secret_storage_mode() != "keyring":
        return "session"

    client, _error = _resolve_keyring_client()
    if client is None:
        return "session"
    try:
        if cleaned:
            client.set_password(KEYRING_SERVICE_NAME, key, cleaned)
        else:
            try:
                client.delete_password(KEYRING_SERVICE_NAME, key)
            except Exception:
                pass
        return "keyring"
    except Exception:
        return "session"
