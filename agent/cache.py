import hashlib
import json

from agent import config


def _file_for(kind, key):
    name = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return config.CACHE_DIR / kind / f"{name}.json"


def load(kind, key):
    path = _file_for(kind, key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def save(kind, key, value):
    path = _file_for(kind, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
