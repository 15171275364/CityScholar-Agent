from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Tuple

try:
    import networkx as nx
except Exception:
    nx = None


class GraphRAG:
    """Graph-augmented RAG helper.

    Responsibilities:
    - Build a document/concept graph from indexed chunks (nodes = chunks/concepts, edges = citation/semantic links).
    - Perform graph-aware retrieval by expanding query nodes and returning context paths.
    - Provide explainable evidence paths for retrieved results.

    Notes: networkx is an optional dependency; if unavailable, methods fall back to simple neighborhood expansion.
    """

    def __init__(self, chunks: Iterable[Any] = (), edges: Iterable[Tuple[str, str]] | None = None):
        self.chunks = list(chunks)
        # map chunk_id -> chunk for quick access
        self._chunk_map = {getattr(c, 'chunk_id', str(i)): c for i, c in enumerate(self.chunks)}
        self.edges = list(edges) if edges is not None else []
        self.graph = None
        if nx is not None:
            self.graph = nx.Graph()
            for c in self.chunks:
                # c is expected to have chunk_id and title attributes
                node_id = getattr(c, "chunk_id", str(len(self.graph)))
                self.graph.add_node(node_id, chunk=c, paper_id=getattr(c, 'paper_id', None))
            for a, b in self.edges:
                self.graph.add_edge(a, b)

    def build_graph(self) -> None:
        """Construct a semantic graph using simple heuristics:
        - Strong edges between chunks of the same paper
        - Semantic edges between chunks with high cosine similarity of embeddings (if provided)

        This method expects that `self.chunks` may have an `_embeddings` attribute set externally
        or that a caller passes embeddings via `build_graph_with_embeddings`.
        """
        if self.graph is None:
            return
        # add strong edges for chunks from same paper
        paper_groups = {}
        for n, data in self.graph.nodes(data=True):
            pid = data.get('paper_id')
            paper_groups.setdefault(pid, []).append(n)
        for pid, nodes in paper_groups.items():
            if not pid:
                continue
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if not self.graph.has_edge(nodes[i], nodes[j]):
                        self.graph.add_edge(nodes[i], nodes[j], weight=1.0, reason='same_paper')
        return

    def build_graph_with_embeddings(self, embeddings, sim_threshold: float = 0.78, max_neighbors: int = 5) -> None:
        """Build semantic edges using cosine similarity on provided embeddings array.
        `embeddings` should be a numpy array aligned with `self.chunks` order.
        """
        if nx is None:
            return
        try:
            import numpy as _np
        except Exception:
            return
        emb = _np.asarray(embeddings, dtype=_np.float32)
        if emb.ndim != 2 or emb.shape[0] != len(self.chunks):
            return
        # normalize
        norms = _np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embn = emb / norms

        for idx, src in enumerate(self.chunks):
            src_id = getattr(src, 'chunk_id')
            vec = embn[idx]
            # compute dot products
            sims = _np.dot(embn, vec)
            # sort indices by similarity, skip self
            order = _np.argsort(-sims)
            neighbors = 0
            for j in order:
                if j == idx:
                    continue
                if neighbors >= max_neighbors:
                    break
                score = float(sims[j])
                if score < sim_threshold:
                    break
                tgt = self.chunks[j]
                tgt_id = getattr(tgt, 'chunk_id')
                if not self.graph.has_edge(src_id, tgt_id):
                    self.graph.add_edge(src_id, tgt_id, weight=score, reason='semantic')
                neighbors += 1

        # ensure same-paper edges exist
        self.build_graph()

    def retrieve(self, query: str, top_k: int = 5) -> List[Any]:
        """Return top_k chunks using graph expansion + base retrieval.

        Returns a list of (chunk, path) tuples where path is a list of node ids connecting evidence.
        """
        if self.graph is None:
            return [(c, []) for c in self.chunks[:top_k]]
        # simple strategy: pick seed nodes by textual match (presence of query) or fallback to highest degree
        seeds = []
        qlow = query.lower()
        for n, data in self.graph.nodes(data=True):
            text = data.get('chunk').text.lower()
            if qlow in text:
                seeds.append(n)
        if not seeds:
            # degree-based seeds
            seeds = sorted(self.graph.nodes, key=lambda x: self.graph.degree[x], reverse=True)[:3]

        # expand seeds to neighbors (radius 1) and collect unique nodes with paths
        collected = {}
        for s in seeds:
            collected[s] = [s]
            for nbr in self.graph.neighbors(s):
                if nbr not in collected:
                    try:
                        path = nx.shortest_path(self.graph, source=s, target=nbr)
                    except Exception:
                        path = [s, nbr]
                    collected[nbr] = path
        # rank by path length and edge weights (prefer shorter paths and higher weights)
        def score_node(nid: str) -> float:
            path = collected.get(nid, [nid])
            length = len(path)
            weight_sum = 0.0
            for a, b in zip(path[:-1], path[1:]):
                w = self.graph.edges[a, b].get('weight', 0.5)
                weight_sum += float(w)
            return weight_sum / (length or 1)

        ranked = sorted(collected.keys(), key=lambda n: score_node(n), reverse=True)[:top_k]
        return [(self.graph.nodes[n]['chunk'], collected.get(n, [])) for n in ranked]

    def explain_path(self, source_id: str, target_id: str) -> List[str]:
        """Return a human-readable path between two nodes if available."""
        if self.graph is None:
            return []
        try:
            path = nx.shortest_path(self.graph, source=source_id, target=target_id)
            return [str(n) for n in path]
        except Exception:
            return []
