"""Thin Redis helpers shared by the Python services.

Not imported by ``shared/__init__.py`` because it depends on the ``redis``
package (the C++ engine and the pure-contract imports must stay dependency
free). Import it explicitly where needed: ``from shared.store import get_redis``.
"""

from __future__ import annotations

import json
from typing import Any

import redis

from . import config


def get_redis() -> "redis.Redis":
    """Return a decoded (str) Redis client using shared config."""
    return redis.Redis(
        host=config.REDIS_HOST,
        port=config.REDIS_PORT,
        db=config.REDIS_DB,
        decode_responses=True,
    )


def set_json(r: "redis.Redis", key: str, value: Any) -> None:
    r.set(key, json.dumps(value))


def get_json(r: "redis.Redis", key: str, default: Any = None) -> Any:
    try:
        raw = r.get(key)
    except redis.RedisError:
        # Redis not up yet (e.g. dashboard started before services) -> degrade.
        return default
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


def push_json_capped(r: "redis.Redis", key: str, value: Any, maxlen: int) -> None:
    """LPUSH a JSON value and trim the list to ``maxlen`` (newest first)."""
    pipe = r.pipeline()
    pipe.lpush(key, json.dumps(value))
    pipe.ltrim(key, 0, maxlen - 1)
    pipe.execute()


def get_json_list(r: "redis.Redis", key: str, count: int) -> list[Any]:
    out = []
    try:
        rows = r.lrange(key, 0, count - 1)
    except redis.RedisError:
        return out
    for raw in rows:
        try:
            out.append(json.loads(raw))
        except (TypeError, json.JSONDecodeError):
            continue
    return out
