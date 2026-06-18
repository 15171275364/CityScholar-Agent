from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def safe_filename(text: str, suffix: str = ".md") -> str:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", text).strip("_")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{cleaned[:40] or 'report'}_{stamp}{suffix}"
