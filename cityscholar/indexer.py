from __future__ import annotations

import pickle
import re
import zlib
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from .models import PaperChunk
from .utils import ensure_dir, normalize_space
import logging
import time

logger = logging.getLogger("cityscholar.indexer")


class PaperIndexer:
    def __init__(self, storage_dir: Path, config=None):
        self.storage_dir = storage_dir
        self.config = config
        self.chunks_path = storage_dir / "chunks.pkl"
        self.vectorizer_path = storage_dir / "vectorizer.pkl"
        self.matrix_path = storage_dir / "tfidf_matrix.pkl"
        self.embeddings_path = storage_dir / "embeddings.npy"
        self.faiss_path = storage_dir / "faiss.index"
        self.faiss_id_map_path = storage_dir / "faiss_id_map.pkl"

    def build(self, papers_dir: Path, chunk_size: int = 900, overlap: int = 120) -> str:
        start = time.perf_counter()
        logger.info(f"Building index from papers in {papers_dir} (chunk_size={chunk_size}, overlap={overlap})")
        ensure_dir(self.storage_dir)
        docs = self._load_documents(papers_dir)
        chunks: list[PaperChunk] = []
        for paper_id, (title, path, pages) in enumerate(docs, 1):
            for page_num, page_text in pages:
                chunks.extend(self._chunk_page(str(paper_id), title, str(path), page_num, page_text, chunk_size, overlap))

        if not chunks:
            raise RuntimeError(f"No readable papers found in {papers_dir}. Please add PDF, TXT, or MD files.")

        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            max_features=60000,
        )
        matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])

        # compute sentence-transformers embeddings if available
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [chunk.text for chunk in chunks]
            embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        except Exception:
            embeddings = None

        with self.chunks_path.open("wb") as file:
            pickle.dump(chunks, file)
        with self.vectorizer_path.open("wb") as file:
            pickle.dump(vectorizer, file)
        with self.matrix_path.open("wb") as file:
            pickle.dump(matrix, file)
        if embeddings is not None:
            try:
                import numpy as _np
                _np.save(self.embeddings_path, embeddings)
                # try to persist a FAISS index for faster startup (optional)
                try:
                    # Only persist FAISS if enabled in config
                    if getattr(self.config, 'faiss_enabled', True):
                        import faiss
                        emb = _np.array(embeddings, dtype=_np.float32)
                        norms = _np.linalg.norm(emb, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        embn = emb / norms
                        index = faiss.IndexFlatIP(embn.shape[1])
                        index.add(embn)
                        # allow override path from config
                        faiss_path = getattr(self.config, 'faiss_path', None) or self.faiss_path
                        faiss.write_index(index, str(faiss_path))
                        # save id map
                        with (self.faiss_id_map_path).open("wb") as f:
                            pickle.dump([c.chunk_id for c in chunks], f)
                except Exception:
                    pass
            except Exception:
                pass
        elapsed = time.perf_counter() - start
        logger.info(f"Built knowledge base: {len(docs)} papers, {len(chunks)} chunks, saved to {self.storage_dir} in {elapsed:.1f}s")
        return f"Built knowledge base: {len(docs)} papers, {len(chunks)} chunks, saved to {self.storage_dir}"

    def load(self) -> tuple[list[PaperChunk], TfidfVectorizer | None, object | None]:
        if not self.chunks_path.exists():
            raise RuntimeError("Local index not found. Run `python main.py build --papers papers` first.")
        with self.chunks_path.open("rb") as file:
            chunks = pickle.load(file)

        vectorizer = None
        matrix = None
        embeddings = None
        if self.vectorizer_path.exists() and self.matrix_path.exists():
            with self.vectorizer_path.open("rb") as file:
                vectorizer = pickle.load(file)
            with self.matrix_path.open("rb") as file:
                matrix = pickle.load(file)
        if self.embeddings_path.exists():
            try:
                import numpy as _np
                embeddings = _np.load(self.embeddings_path)
            except Exception:
                embeddings = None
        faiss_index = None
        faiss_id_map = None
        faiss_path_to_load = getattr(self.config, 'faiss_path', None) or self.faiss_path
        if getattr(self.config, 'faiss_enabled', True) and Path(faiss_path_to_load).exists():
            try:
                import faiss
                faiss_index = faiss.read_index(str(faiss_path_to_load))
                if self.faiss_id_map_path.exists():
                    with self.faiss_id_map_path.open('rb') as f:
                        faiss_id_map = pickle.load(f)
            except Exception:
                faiss_index = None
                faiss_id_map = None
        return chunks, vectorizer, matrix, embeddings, faiss_index, faiss_id_map

    def _load_documents(self, papers_dir: Path) -> list[tuple[str, Path, list[tuple[int | None, str]]]]:
        if not papers_dir.exists():
            raise RuntimeError(f"Papers directory does not exist: {papers_dir}")
        files = sorted(
            path for path in papers_dir.iterdir()
            if path.suffix.lower() in {".pdf", ".txt", ".md"} and path.is_file()
        )
        docs = []
        for path in files:
            pages = self._read_pdf(path) if path.suffix.lower() == ".pdf" else [(None, path.read_text(encoding="utf-8", errors="ignore"))]
            pages = [(page, normalize_space(text)) for page, text in pages if normalize_space(text)]
            if pages:
                docs.append((path.stem, path, pages))
        return docs

    def _read_pdf(self, path: Path) -> list[tuple[int | None, str]]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            return self._read_pdf_fallback(path)

        reader = PdfReader(str(path))
        pages = []
        for idx, page in enumerate(reader.pages, 1):
            pages.append((idx, page.extract_text() or ""))
        return pages

    def _read_pdf_fallback(self, path: Path) -> list[tuple[int | None, str]]:
        """Best-effort PDF text extraction used when pypdf is unavailable."""
        data = path.read_bytes()
        texts: list[str] = []
        for index, match in enumerate(re.finditer(rb"stream\r?\n(.*?)\r?\nendstream", data, re.S)):
            if index >= 12:
                break
            stream = match.group(1)
            if len(stream) > 2_000_000:
                continue
            candidates = [stream]
            try:
                candidates.append(zlib.decompress(stream))
            except Exception:
                pass
            for candidate in candidates:
                extracted = self._extract_pdf_text_tokens(candidate)
                if extracted:
                    texts.append(extracted)

        merged = normalize_space(" ".join(texts))
        if len(merged) > 80:
            return [(None, merged)]

        title_text = path.stem.replace("_", " ").replace("-", " ")
        return [(None, self._metadata_from_filename(title_text))]

    def _extract_pdf_text_tokens(self, data: bytes) -> str:
        chunks: list[str] = []
        for token in re.findall(rb"\((?:\\.|[^\\)])*\)", data):
            raw = token[1:-1]
            raw = raw.replace(rb"\(", b"(").replace(rb"\)", b")").replace(rb"\\n", b" ")
            text = raw.decode("utf-8", errors="ignore") or raw.decode("latin-1", errors="ignore")
            if text and sum(ch.isalnum() for ch in text) >= 2:
                chunks.append(text)
        return normalize_space(" ".join(chunks))

    def _metadata_from_filename(self, title_text: str) -> str:
        lower = title_text.lower()
        terms = []
        keyword_map = {
            "federated learning": "federated learning 联邦学习",
            "privacy": "privacy preserving differential privacy 隐私保护 差分隐私",
            "personalized": "personalized federated learning 个性化联邦学习",
            "non-iid": "non-iid heterogeneous data 非独立同分布 异构数据",
            "heterogeneous": "heterogeneous clients data 异构客户端 异构数据",
            "robust": "robust aggregation attack defense 鲁棒聚合 攻击 防御",
            "aggregation": "aggregation 聚合算法",
            "survey": "survey review 综述",
            "review": "review survey 综述",
            "graph": "graph federated learning 图学习",
            "blockchain": "blockchain 区块链",
        }
        for keyword, value in keyword_map.items():
            if keyword in lower or keyword.replace("-", " ") in lower:
                terms.append(value)
        return (
            f"{title_text}. This local PDF is indexed from filename metadata because no PDF parser is installed. "
            f"Potential topics: {'; '.join(terms) if terms else 'federated learning, privacy computing, review, methods, applications'}."
        )

    def _chunk_page(
        self,
        paper_id: str,
        title: str,
        source_path: str,
        page: int | None,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[PaperChunk]:
        chunks = []
        start = 0
        safe_overlap = min(overlap, chunk_size // 2)
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if len(chunk_text) > 40:
                chunks.append(
                    PaperChunk(
                        chunk_id=f"{paper_id}-{page or 0}-{len(chunks) + 1}",
                        paper_id=paper_id,
                        title=title,
                        source_path=source_path,
                        page=page,
                        text=chunk_text,
                    )
                )
            if end == len(text):
                break
            start = end - safe_overlap
        return chunks

    def rebuild_faiss(self, sim_threshold: float = 0.0) -> str:
        """Rebuild FAISS index from existing embeddings and save to disk.

        Returns a status message.
        """
        logger.info("Rebuilding FAISS index from embeddings...")
        try:
            import numpy as _np
        except Exception:
            return "NumPy not available; cannot rebuild FAISS index."

        if not self.embeddings_path.exists():
            return "Embeddings file not found; run build first to generate embeddings."

        try:
            emb = _np.load(self.embeddings_path)
        except Exception:
            return "Failed to load embeddings array."

        try:
            import faiss
            emb = _np.array(emb, dtype=_np.float32)
            norms = _np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            embn = emb / norms
            index = faiss.IndexFlatIP(embn.shape[1])
            index.add(embn)
            faiss_path = getattr(self, 'faiss_path', None) or self.faiss_path
            faiss.write_index(index, str(faiss_path))
            # attempt to rebuild id map from chunks file
            if self.chunks_path.exists():
                with self.chunks_path.open('rb') as f:
                    import pickle as _p
                    chunks = _p.load(f)
                with self.faiss_id_map_path.open('wb') as f:
                    import pickle as _p
                    _p.dump([c.chunk_id for c in chunks], f)
            logger.info(f"Rebuilt FAISS index and saved to {faiss_path}")
            return f"Rebuilt FAISS index and saved to {faiss_path}"
        except Exception as exc:
            logger.exception("Failed to rebuild FAISS index")
            return f"Failed to rebuild FAISS index: {exc}"
