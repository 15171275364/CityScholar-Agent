from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_local_env(root: Path) -> None:
    env_path = root / ".env.local"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class AppConfig:
    storage_dir: Path
    outputs_dir: Path
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_model: str = "qwen-plus"
    # FAISS and GraphRAG related configuration
    faiss_enabled: bool = True
    graph_rag_enabled: bool = True
    graph_sim_threshold: float = 0.78
    graph_max_neighbors: int = 6
    faiss_path: Path | None = None
    # LLM client tuning
    llm_retries: int = 2
    llm_timeout: int = 45
    llm_backoff_base: float = 1.0
    # Intelligence enhancement
    max_reflection_rounds: int = 1      # self-reflection rounds (0 = disabled)
    enable_query_expansion: bool = True  # expand queries with synonyms/related terms
    enable_rerank: bool = True           # cross-chunk reranking after retrieval
    rrf_k: int = 60                      # Reciprocal Rank Fusion constant

    @classmethod
    def from_env(cls, storage_dir: Path) -> "AppConfig":
        root = storage_dir.parent if storage_dir.parent != Path("") else Path(".")
        load_local_env(root)
        return cls(
            storage_dir=storage_dir,
            outputs_dir=root / "outputs",
            llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("API_KEY"),
            llm_base_url=os.getenv("LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode"),
            llm_model=os.getenv("LLM_MODEL", "qwen-plus"),
            faiss_enabled=os.getenv("FAISS_ENABLED", "1") not in ("0", "false", "False"),
            graph_rag_enabled=os.getenv("GRAPH_RAG_ENABLED", "1") not in ("0", "false", "False"),
            graph_sim_threshold=float(os.getenv("GRAPH_SIM_THRESHOLD", "0.78")),
            graph_max_neighbors=int(os.getenv("GRAPH_MAX_NEIGHBORS", "6")),
            faiss_path=(Path(os.getenv("FAISS_PATH")) if os.getenv("FAISS_PATH") else None),
            llm_retries=int(os.getenv("LLM_RETRIES", "2")),
            llm_timeout=int(os.getenv("LLM_TIMEOUT", "45")),
            llm_backoff_base=float(os.getenv("LLM_BACKOFF_BASE", "1.0")),
            max_reflection_rounds=int(os.getenv("MAX_REFLECTION_ROUNDS", "1")),
            enable_query_expansion=os.getenv("ENABLE_QUERY_EXPANSION", "1") not in ("0", "false", "False"),
            enable_rerank=os.getenv("ENABLE_RERANK", "1") not in ("0", "false", "False"),
            rrf_k=int(os.getenv("RRF_K", "60")),
        )
