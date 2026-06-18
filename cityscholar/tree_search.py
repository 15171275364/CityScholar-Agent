"""推理树搜索模块 — 多步推理路径探索与最优选择。

核心能力：
- ReasoningNode: 推理节点，包含推理状态和评估分数
- 推理路径展开：基于 LLM 或启发式规则生成子节点
- Best-first 搜索：优先探索高评分节点
- 多路径评估：综合路径长度、推理质量、证据覆盖度
- 与 Agent 集成：用于复杂问题的多步推理
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from heapq import heappush, heappop
from typing import Any, Callable

logger = logging.getLogger("cityscholar.tree_search")


@dataclass
class ReasoningNode:
    """推理树中的一个节点。"""
    state: str                    # 当前推理状态/中间结论
    depth: int = 0                # 搜索深度
    score: float = 0.0            # 综合评分
    parent: ReasoningNode | None = None
    evidence_used: list[str] = field(default_factory=list)
    reasoning_step: str = ""      # 从父节点到当前节点的推理步骤
    is_terminal: bool = False     # 是否为终止节点

    @property
    def path(self) -> list[str]:
        """从根到当前节点的完整推理路径。"""
        node = self
        path = []
        while node:
            path.append(node.reasoning_step or node.state[:100])
            node = node.parent
        return list(reversed(path))

    @property
    def path_length(self) -> int:
        node = self
        length = 0
        while node:
            length += 1
            node = node.parent
        return length

    def __lt__(self, other: ReasoningNode) -> bool:
        return self.score > other.score  # 最大堆


class TreeSearchReasoner:
    """推理树搜索器：通过多路径探索找到最优推理链。"""

    def __init__(self, max_depth: int = 3, max_nodes: int = 30,
                 beam_width: int = 3, min_score_threshold: float = 0.2):
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.beam_width = beam_width
        self.min_score_threshold = min_score_threshold
        self._expand_count = 0
        self._eval_count = 0

    def search(self, root_state: str,
               expand_fn: Callable[[ReasoningNode], list[ReasoningNode]],
               score_fn: Callable[[ReasoningNode], float],
               terminal_fn: Callable[[ReasoningNode], bool] | None = None,
               max_time: float = 30.0) -> ReasoningNode:
        """执行 Best-first 推理树搜索。

        Args:
            root_state: 初始推理状态（如用户问题）
            expand_fn: 展开函数，给定节点返回子节点列表
            score_fn: 评分函数，给定节点返回 0~1 的分数
            terminal_fn: 终止判断函数，返回 True 则停止展开
            max_time: 最大搜索时间（秒）
        """
        start = time.perf_counter()
        self._expand_count = 0
        self._eval_count = 0

        root = ReasoningNode(state=root_state, score=score_fn(root_state))
        self._eval_count += 1

        best = root
        heap: list[ReasoningNode] = []
        heappush(heap, root)
        explored: list[ReasoningNode] = [root]

        while heap and len(explored) < self.max_nodes:
            if time.perf_counter() - start > max_time:
                logger.info(f"搜索超时（{max_time}s），已探索 {len(explored)} 个节点")
                break

            current = heappop(heap)

            # 更新最优
            if current.score > best.score:
                best = current

            # 终止条件
            if current.is_terminal:
                continue
            if terminal_fn and terminal_fn(current):
                current.is_terminal = True
                if current.score > best.score:
                    best = current
                continue
            if current.depth >= self.max_depth:
                continue

            # 展开子节点
            try:
                children = expand_fn(current)
                self._expand_count += 1
            except Exception as e:
                logger.warning(f"展开失败：{e}")
                continue

            # 评分并剪枝
            scored_children = []
            for child in children:
                child.score = score_fn(child)
                self._eval_count += 1
                child.parent = current
                child.depth = current.depth + 1
                if child.score >= self.min_score_threshold:
                    scored_children.append(child)

            # Beam 剪枝：保留评分最高的 beam_width 个
            scored_children.sort(key=lambda n: n.score, reverse=True)
            for child in scored_children[:self.beam_width]:
                heappush(heap, child)
                explored.append(child)

        elapsed = time.perf_counter() - start
        logger.info(
            f"树搜索完成：深度={best.path_length}, "
            f"展开={self._expand_count}, 评估={self._eval_count}, "
            f"节点={len(explored)}, 耗时={elapsed:.2f}s"
        )
        return best

    def to_reasoning_trace(self, best_node: ReasoningNode) -> str:
        """将最优推理路径转化为可读的推理追踪文本。"""
        path = best_node.path
        if not path:
            return "无推理路径。"

        lines = ["## 推理路径\n"]
        for idx, step in enumerate(path):
            prefix = "→" if idx < len(path) - 1 else "∴"
            lines.append(f"  {'  ' * idx}{prefix} {step}")

        lines.append(f"\n**最终评分：** {best_node.score:.3f}")
        lines.append(f"**搜索统计：** 展开 {self._expand_count} 次，评估 {self._eval_count} 次")
        return "\n".join(lines)


# ── 与 Agent 集成的推理扩展 ──────────────────────────────────────────────────

class AgentTreeSearch:
    """将树搜索与 CityScholar Agent 集成，用于复杂问题的多步推理。"""

    def __init__(self, agent=None, llm_client=None):
        self.agent = agent
        self.llm = llm_client
        self.searcher = TreeSearchReasoner(max_depth=3, max_nodes=20, beam_width=3)

    def reason(self, question: str, top_k: int = 5) -> dict:
        """对复杂问题执行多步推理树搜索。"""
        # 定义展开函数：基于当前推理状态生成子问题/子推理
        def expand(node: ReasoningNode) -> list[ReasoningNode]:
            children = []
            # 生成 2-3 个子推理方向
            sub_questions = self._decompose(node.state)
            for sub_q in sub_questions:
                child = ReasoningNode(
                    state=sub_q,
                    reasoning_step=f"子问题：{sub_q}",
                )
                children.append(child)
            return children

        # 定义评分函数：综合考虑证据覆盖度和推理深度
        def score(node: ReasoningNode) -> float:
            if not self.agent:
                return 0.5
            try:
                retriever = self.agent._retriever()
                results = retriever.search(node.state, top_k=3)
                if not results:
                    return 0.1
                # 证据相关性得分
                avg_score = sum(r.score for r in results) / len(results)
                # 深度惩罚（避免过深）
                depth_penalty = max(0, 1.0 - node.depth * 0.15)
                # 新颖性奖励（如果证据中有新论文）
                existing = set(node.evidence_used)
                new_papers = {r.chunk.title for r in results if r.chunk.title not in existing}
                novelty_bonus = min(0.2, len(new_papers) * 0.05)

                return min(1.0, avg_score * depth_penalty + novelty_bonus)
            except Exception:
                return 0.3

        # 定义终止函数
        def terminal(node: ReasoningNode) -> bool:
            return node.depth >= 2  # 两层深度即可

        # 执行搜索
        best = self.searcher.search(
            root_state=question,
            expand_fn=expand,
            score_fn=score,
            terminal_fn=terminal,
        )

        # 基于最优路径生成最终回答
        trace = self.searcher.to_reasoning_trace(best)
        final_answer = self._synthesize_answer(question, best)

        return {
            "answer": final_answer,
            "reasoning_trace": trace,
            "best_score": best.score,
            "search_depth": best.path_length,
        }

    def _decompose(self, question: str) -> list[str]:
        """将问题分解为子问题。"""
        if self.llm and self.llm.enabled:
            result = self.llm.generate_json(
                f"将以下问题分解为 2-3 个子问题：\n{question}",
                {"sub_questions": [question]},
                "decompose",
                max_tokens=300,
            )
            return result.get("sub_questions", [question])[:3]
        # 简单启发式分解
        return [
            f"{question}的研究背景是什么？",
            f"{question}的主要方法有哪些？",
            f"{question}的实验结果如何？",
        ][:2]

    def _synthesize_answer(self, question: str, best_node: ReasoningNode) -> str:
        """基于最优推理路径合成最终回答。"""
        path = best_node.path
        if len(path) <= 1:
            return best_node.state

        steps = "\n".join(f"- {step}" for step in path)
        prompt = (
            f"基于以下推理路径，回答问题：{question}\n\n"
            f"推理路径：\n{steps}\n\n"
            "请给出综合性的回答。"
        )
        if self.llm and self.llm.enabled:
            return self.llm.generate(prompt, f"基于推理路径的回答：\n{steps}", task="answer")

        return f"## 推理路径\n\n{steps}\n\n**综合回答：** 基于以上推理路径，对问题进行综合分析。"
