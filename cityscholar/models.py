from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PaperChunk:
    chunk_id: str
    paper_id: str
    title: str
    source_path: str
    page: int | None
    text: str

    def citation(self) -> str:
        page = f", page {self.page}" if self.page else ""
        return f"{self.title}{page}"


@dataclass
class SearchResult:
    chunk: PaperChunk
    score: float
    keyword_score: float = 0.0
    vector_score: float = 0.0
    path: list[str] = field(default_factory=list)


@dataclass
class AgentResponse:
    title: str
    content: str
    evidence: list[SearchResult] = field(default_factory=list)
    confidence: float = 0.0        # 0.0 ~ 1.0 self-assessed confidence
    reasoning_trace: str = ""      # step-by-step reasoning log

    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", self.content.strip()]
        if self.confidence > 0:
            lines.extend(["", f"**置信度：** {self.confidence:.0%}"])
        if self.reasoning_trace:
            lines.extend(["", "<details><summary>推理过程</summary>", "", self.reasoning_trace.strip(), "", "</details>"])
        if self.evidence:
            lines.extend(["", "## Evidence"])
            for idx, item in enumerate(self.evidence, 1):
                snippet = item.chunk.text.replace("\n", " ")[:260]
                lines.append(f"{idx}. **{item.chunk.citation()}** (score={item.score:.3f})")
                lines.append(f"   {snippet}")
        return "\n".join(lines).strip()
