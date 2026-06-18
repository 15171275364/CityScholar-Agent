from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any
import logging
import time

logger = logging.getLogger("cityscholar.memory")


@dataclass
class MemoryEntry:
    type: str
    content: str
    metadata: dict | None = None
    ts: str = datetime.utcnow().isoformat()


class MemoryStore:
    """A tiny persistent JSONL memory store for agent artifacts (analyses, Q/A, outlines).

    Usage:
      mem = MemoryStore(storage_dir)
      mem.add('analysis', content, {'paper': 'title'})
      entries = mem.query('keyword')
    """

    def __init__(self, storage_dir: Path):
        self.path = storage_dir / "memory.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, type: str, content: str, metadata: dict | None = None) -> None:
        entry = MemoryEntry(type=type, content=content, metadata=metadata or {})
        t0 = time.perf_counter()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        logger.info(f"Memory added type={type} size={len(content):,} chars in {time.perf_counter()-t0:.3f}s")

    def all(self) -> list[MemoryEntry]:
        if not self.path.exists():
            return []
        out: list[MemoryEntry] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    out.append(MemoryEntry(**obj))
                except Exception:
                    continue
        return out

    def query(self, keyword: str, limit: int = 10) -> list[MemoryEntry]:
        kw = keyword.lower()
        hits: list[MemoryEntry] = []
        for e in self.all():
            if kw in e.content.lower() or any(kw in str(v).lower() for v in (e.metadata or {}).values()):
                hits.append(e)
                if len(hits) >= limit:
                    break
        return hits
