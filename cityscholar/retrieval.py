from __future__ import annotations

import math
import re
import time
import logging
from collections import Counter
from typing import Iterable

from sklearn.metrics.pairwise import cosine_similarity

from .models import PaperChunk, SearchResult
from .utils import tokenize
from .graph_rag import GraphRAG

logger = logging.getLogger("cityscholar.retrieval")


# ── Query expansion: academic synonyms & related terms ─────────────────────────

_ACADEMIC_EXPANSIONS: dict[str, list[str]] = {
    # Chinese academic synonyms
    "方法": ["approach", "methodology", "technique", "algorithm"],
    "模型": ["model", "framework", "architecture"],
    "实验": ["experiment", "evaluation", "benchmark", "empirical"],
    "结果": ["results", "findings", "outcomes"],
    "贡献": ["contribution", "novelty", "advancement"],
    "局限": ["limitation", "shortcoming", "weakness", "gap"],
    "数据": ["dataset", "corpus", "data"],
    "优化": ["optimization", "improvement", "enhancement"],
    "隐私": ["privacy", "confidentiality", "differential privacy"],
    "联邦学习": ["federated learning", "FL", "collaborative learning"],
    "综述": ["survey", "review", "overview", "state-of-the-art"],
    "应用场景": ["application", "use case", "scenario", "deployment"],
    "未来方向": ["future work", "research direction", "open problem"],
    "研究现状": ["state-of-the-art", "current progress", "literature"],
    "大语言模型": ["LLM", "large language model", "GPT", "foundation model"],
    "智能体": ["agent", "intelligent agent", "autonomous agent"],
    "图神经网络": ["GNN", "graph neural network", "graph learning"],
    "检索增强": ["RAG", "retrieval-augmented", "knowledge retrieval"],
    # English academic synonyms
    "approach": ["method", "methodology", "technique", "framework"],
    "experiment": ["evaluation", "benchmark", "empirical study", "validation"],
    "result": ["finding", "outcome", "conclusion", "observation"],
    "limitation": ["weakness", "shortcoming", "gap", "challenge"],
    "method": ["approach", "technique", "algorithm", "procedure"],
    "model": ["framework", "architecture", "system"],
    "improve": ["enhance", "optimize", "advance", "refine"],
    "compare": ["contrast", "evaluate", "benchmark", "assess"],
    "survey": ["review", "overview", "synthesis", "meta-analysis"],
}


def _expand_query(query: str, max_terms: int = 8) -> str:
    """Expand the query with related academic terms to improve recall."""
    tokens = tokenize(query)
    query_lower = query.lower()
    expansion_terms = []

    for token in tokens:
        if token in _ACADEMIC_EXPANSIONS:
            for term in _ACADEMIC_EXPANSIONS[token]:
                if term.lower() not in query_lower:
                    expansion_terms.append(term)
                    if len(expansion_terms) >= max_terms:
                        break
        if len(expansion_terms) >= max_terms:
            break

    if expansion_terms:
        expanded = query + " " + " ".join(expansion_terms)
        logger.info(f"Query expanded: '{query[:50]}...' + {expansion_terms}")
        return expanded
    return query


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def _rrf_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        rankings: list of ranked lists, each containing chunk_ids
        k: RRF constant (higher = less weight on top ranks)

    Returns:
        sorted list of (chunk_id, rrf_score)
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Reranking ──────────────────────────────────────────────────────────────────

def _rerank_results(query: str, results: list[SearchResult],
                    vectorizer=None, matrix=None,
                    chunks: list[PaperChunk] | None = None) -> list[SearchResult]:
    """Rerank results using a query-centric scoring heuristic.

    Combines original score with:
    - Title match bonus
    - Length normalization (penalize very short/long chunks)
    - Position diversity (avoid too many from same paper)
    - Keyword density in chunk
    """
    if not results:
        return results

    query_tokens = set(tokenize(query))
    seen_papers: dict[str, int] = {}

    reranked = []
    for item in results:
        bonus = 0.0

        # 1. Title match bonus (0 ~ +0.15)
        title_tokens = set(tokenize(item.chunk.title))
        title_overlap = len(query_tokens & title_tokens) / max(1, len(query_tokens))
        bonus += 0.15 * min(1.0, title_overlap * 2)

        # 2. Length normalization: prefer chunks of moderate length
        text_len = len(item.chunk.text)
        if 200 < text_len < 1500:
            bonus += 0.05
        elif text_len < 100:
            bonus -= 0.05

        # 3. Keyword density: ratio of query terms found in chunk
        chunk_tokens = set(tokenize(item.chunk.text))
        density = len(query_tokens & chunk_tokens) / max(1, len(query_tokens))
        bonus += 0.1 * density

        # 4. Diversity: penalize over-representation from same paper
        paper = item.chunk.title
        seen_papers[paper] = seen_papers.get(paper, 0) + 1
        if seen_papers[paper] > 2:
            bonus -= 0.03 * (seen_papers[paper] - 2)

        new_score = max(0.0, item.score + bonus)
        reranked.append(SearchResult(
            chunk=item.chunk,
            score=new_score,
            keyword_score=item.keyword_score,
            vector_score=item.vector_score,
            path=item.path,
        ))

    return sorted(reranked, key=lambda x: x.score, reverse=True)


# ── Hybrid Retriever ───────────────────────────────────────────────────────────

class HybridRetriever:
    def __init__(self, chunks: list[PaperChunk], vectorizer=None, matrix=None,
                 embeddings=None, faiss_index=None, faiss_id_map=None, config=None):
        self.chunks = chunks
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.embeddings = embeddings
        self._config = config
        self._faiss_index = None
        self._faiss_id_map = None
        self.graph_rag = None

        # Build FAISS index if embeddings are available
        if self.embeddings is not None:
            try:
                import numpy as _np
                import faiss
                if faiss_index is not None:
                    self._faiss_index = faiss_index
                    self._faiss_id_map = faiss_id_map if faiss_id_map is not None else [c.chunk_id for c in self.chunks]
                else:
                    emb = _np.array(self.embeddings, dtype=_np.float32)
                    norms = _np.linalg.norm(emb, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0
                    emb = emb / norms
                    index = faiss.IndexFlatIP(emb.shape[1])
                    index.add(emb)
                    self._faiss_index = index
                    self._faiss_id_map = [c.chunk_id for c in self.chunks]
            except Exception:
                self._faiss_index = None
                self._faiss_id_map = None

        # Build semantic graph for GraphRAG
        try:
            if self.embeddings is not None and getattr(config, "graph_rag_enabled", True):
                self.graph_rag = GraphRAG(self.chunks)
                sim_th = getattr(config, "graph_sim_threshold", 0.78)
                max_n = getattr(config, "graph_max_neighbors", 6)
                self.graph_rag.build_graph_with_embeddings(
                    self.embeddings, sim_threshold=sim_th, max_neighbors=max_n
                )
        except Exception:
            self.graph_rag = None

    # ── Main search entry point ────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5,
               paper_filter: list[str] | None = None,
               expand: bool | None = None) -> list[SearchResult]:
        """Hybrid search with optional query expansion, RRF fusion, and reranking."""
        selected = self._filter_chunks(paper_filter)
        if not selected:
            return []

        start = time.perf_counter()

        # Step 1: Optionally expand query for better recall
        use_expansion = expand if expand is not None else getattr(self._config, "enable_query_expansion", True)
        search_queries = [query]
        if use_expansion:
            expanded = _expand_query(query)
            if expanded != query:
                search_queries.append(expanded)

        # Step 2: Compute keyword and vector scores for each search query
        all_keyword_scores: list[list[float]] = []
        all_vector_scores: list[dict[str, float]] = []

        for sq in search_queries:
            all_keyword_scores.append(self._keyword_scores(sq, selected))
            all_vector_scores.append(self._vector_scores(sq, selected))

        # Step 3: Reciprocal Rank Fusion for combining multiple query results
        rrf_k = getattr(self._config, "rrf_k", 60) if self._config else 60

        # Build rank lists for keyword and vector
        chunk_ids = [c.chunk_id for c in selected]

        # Merge keyword scores (take max across queries)
        merged_keyword: dict[str, float] = {}
        for ks_list in all_keyword_scores:
            for local_idx, score in enumerate(ks_list):
                cid = chunk_ids[local_idx]
                merged_keyword[cid] = max(merged_keyword.get(cid, 0.0), score)

        # Merge vector scores (take max across queries)
        merged_vector: dict[str, float] = {}
        for vs_dict in all_vector_scores:
            for cid, score in vs_dict.items():
                merged_vector[cid] = max(merged_vector.get(cid, 0.0), score)

        # Step 4: Compute final scores using weighted fusion
        results = []
        for chunk in selected:
            kw = merged_keyword.get(chunk.chunk_id, 0.0)
            vec = merged_vector.get(chunk.chunk_id, 0.0)
            # Dynamic weight: give more weight to vector when query is long/complex
            word_count = len(tokenize(query))
            if word_count > 8:
                vec_w, kw_w = 0.65, 0.35  # complex query → trust semantic more
            else:
                vec_w, kw_w = 0.55, 0.45  # simple query → balanced
            score = kw_w * kw + vec_w * vec
            if score > 0:
                results.append(SearchResult(chunk, score, kw, vec))

        if not results and selected:
            results = [SearchResult(chunk, 0.0) for chunk in selected[:top_k]]

        # Step 5: GraphRAG expansion
        try:
            if self.graph_rag is not None and results:
                seed_ids = [r.chunk.chunk_id for r in results[:3]]
                expanded_graph = {}
                for sid in seed_ids:
                    neighbors = self.graph_rag.retrieve(query, top_k=top_k * 3)
                    for c, path in neighbors:
                        expanded_graph[c.chunk_id] = (c, path)
                exist_ids = {r.chunk.chunk_id for r in results}
                for cid, (chunk, path) in expanded_graph.items():
                    if cid not in exist_ids:
                        v = merged_vector.get(cid, 0.0)
                        score = 0.6 * v  # conservative boost for graph-expanded nodes
                        results.append(SearchResult(chunk, score, 0.0, v, path))
        except Exception:
            pass

        # Step 6: Rerank
        use_rerank = getattr(self._config, "enable_rerank", True) if self._config else True
        if use_rerank and len(results) > top_k:
            results = _rerank_results(query, results, self.vectorizer, self.matrix, self.chunks)

        elapsed = time.perf_counter() - start
        logger.info(
            f"search(query='{query[:80]}...', top_k={top_k}) → {len(results)} results "
            f"in {elapsed:.3f}s (expansion={use_expansion}, rerank={use_rerank})"
        )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    # ── Paper lookup ───────────────────────────────────────────────────────────

    def find_papers(self, keywords: list[str]) -> dict[str, list[PaperChunk]]:
        """Find papers matching keywords with multi-strategy matching.

        Returns ALL matching papers (not just the first), ranked by match quality.
        Strategies: exact title > author+year > partial title > semantic search.
        """
        matched: dict[str, list[PaperChunk]] = {}
        titles = sorted({chunk.title for chunk in self.chunks})

        for keyword in keywords:
            lower = keyword.lower().strip()
            if not lower:
                continue

            # Strategy 1: exact substring in title (highest confidence)
            exact_matches = [t for t in titles if lower in t.lower()]
            for t in exact_matches:
                if t not in matched:
                    matched[t] = [c for c in self.chunks if c.title == t]

            # Strategy 2: fuzzy match — match individual tokens
            if not exact_matches:
                kw_tokens = set(tokenize(keyword))
                scored_titles = []
                for t in titles:
                    t_tokens = set(tokenize(t))
                    overlap = len(kw_tokens & t_tokens)
                    if overlap > 0:
                        scored_titles.append((overlap / max(1, len(kw_tokens)), t))
                scored_titles.sort(key=lambda x: x[0], reverse=True)
                for score, t in scored_titles[:5]:  # top 5 fuzzy matches
                    if t not in matched:
                        matched[t] = [c for c in self.chunks if c.title == t]

            # Strategy 3: semantic search fallback
            if not exact_matches and not scored_titles:
                sr = self.search(keyword, top_k=3)
                for r in sr:
                    t = r.chunk.title
                    if t not in matched:
                        matched[t] = [c for c in self.chunks if c.title == t]

        return matched

    def list_all_papers(self) -> list[str]:
        """Return a sorted list of all unique paper titles in the index."""
        return sorted({chunk.title for chunk in self.chunks})

    # ── Internal scoring methods ───────────────────────────────────────────────

    def _filter_chunks(self, paper_filter: list[str] | None) -> list[PaperChunk]:
        if not paper_filter:
            return self.chunks
        filters = [item.lower() for item in paper_filter]
        return [chunk for chunk in self.chunks
                if any(item in chunk.title.lower() for item in filters)]

    def _keyword_scores(self, query: str, chunks: list[PaperChunk]) -> list[float]:
        """Enhanced keyword scoring with BM25-like tf normalization."""
        query_terms = tokenize(query)
        if not query_terms:
            return [0.0 for _ in chunks]
        query_counter = Counter(query_terms)

        # BM25-inspired scoring parameters
        avg_dl = sum(len(tokenize(c.text)) for c in chunks) / max(1, len(chunks))
        k1, b = 1.5, 0.75

        scores = []
        for chunk in chunks:
            chunk_counter = Counter(tokenize(chunk.text))
            dl = sum(chunk_counter.values())
            score = 0.0
            for term, qtf in query_counter.items():
                if term in chunk_counter:
                    tf = chunk_counter[term]
                    tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / max(1, avg_dl)))
                    # IDF approximation: log(N / df), using simple presence
                    idf = 1.0  # simplified; could be computed corpus-wide
                    score += qtf * tf_norm * idf
            scores.append(score)

        max_score = max(scores) if scores else 0.0
        return [s / max_score if max_score else 0.0 for s in scores]

    def _vector_scores(self, query: str, chunks: list[PaperChunk]) -> dict[str, float]:
        """Compute vector similarity scores. Prefers dense embeddings, falls back to TF-IDF."""
        if self.embeddings is not None:
            try:
                import numpy as _np
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    qvec = model.encode([query], convert_to_numpy=True)[0]
                except Exception:
                    qvec = None

                if qvec is not None:
                    # FAISS fast path
                    if self._faiss_index is not None:
                        q = _np.array(qvec, dtype=_np.float32)
                        q = q / (_np.linalg.norm(q) or 1.0)
                        D, I = self._faiss_index.search(q.reshape(1, -1), k=len(self._faiss_id_map))
                        raw = {}
                        wanted_ids = {c.chunk_id for c in chunks}
                        max_score = 0.0
                        for score, idx in zip(D[0], I[0]):
                            if idx < 0:
                                continue
                            cid = self._faiss_id_map[idx]
                            if cid in wanted_ids:
                                raw[cid] = float(score)
                                max_score = max(max_score, raw[cid])
                        return {cid: s / max_score if max_score else 0.0
                                for cid, s in raw.items()}

                    # Direct cosine similarity
                    all_scores = _np.dot(self.embeddings, qvec) / (
                        _np.linalg.norm(self.embeddings, axis=1)
                        * (_np.linalg.norm(qvec) or 1e-12)
                    )
                    wanted_ids = {c.chunk_id for c in chunks}
                    raw = {self.chunks[i].chunk_id: float(all_scores[i])
                           for i in range(len(self.chunks))
                           if self.chunks[i].chunk_id in wanted_ids}
                    max_score = max(raw.values()) if raw else 0.0
                    return {cid: s / max_score if max_score else 0.0
                            for cid, s in raw.items()}
            except Exception:
                pass

        # TF-IDF fallback
        if self.vectorizer is None or self.matrix is None:
            return {}
        try:
            query_vector = self.vectorizer.transform([query])
            all_scores = cosine_similarity(query_vector, self.matrix).ravel()
        except Exception:
            return {}

        wanted_ids = {c.chunk_id for c in chunks}
        raw = {self.chunks[i].chunk_id: float(all_scores[i])
               for i in range(len(self.chunks))
               if self.chunks[i].chunk_id in wanted_ids}
        max_score = max(raw.values()) if raw else 0.0
        return {cid: s / max_score if max_score else 0.0 for cid, s in raw.items()}
