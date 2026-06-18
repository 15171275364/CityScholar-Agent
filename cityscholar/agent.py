from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path

from .config import AppConfig
from .indexer import PaperIndexer
from .llm import LLMClient, evidence_text
from .models import AgentResponse, SearchResult
from .retrieval import HybridRetriever
from .utils import ensure_dir, safe_filename
from .memory import MemoryStore

logger = logging.getLogger("cityscholar.agent")


class CityScholarAgent:
    def __init__(self, config: AppConfig):
        self.config = config
        self.indexer = PaperIndexer(config.storage_dir, config)
        self.llm = LLMClient(config)
        try:
            self.memory = MemoryStore(config.storage_dir)
        except Exception:
            self.memory = None

    # ════════════════════════════════════════════════════════════════════════════
    # Public API
    # ════════════════════════════════════════════════════════════════════════════

    def build(self, papers_dir: Path, chunk_size: int = 900, overlap: int = 120) -> str:
        start = time.perf_counter()
        logger.info(f"Starting index build for {papers_dir}")
        res = self.indexer.build(papers_dir, chunk_size, overlap)
        logger.info(f"Index build finished in {time.perf_counter()-start:.1f}s")
        return res

    def answer(self, question: str, top_k: int = 5) -> AgentResponse:
        """Answer a question with multi-step reasoning and self-reflection."""
        start = time.perf_counter()
        reasoning_steps = []

        # ── Step 1: Query decomposition (for complex questions) ──
        sub_queries = self._decompose_query(question)
        reasoning_steps.append(f"查询分解：{' | '.join(sub_queries)}")

        # ── Step 2: Multi-query retrieval with expansion ──
        retriever = self._retriever()
        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()

        for sq in sub_queries:
            results = retriever.search(sq, top_k=top_k)
            for r in results:
                if r.chunk.chunk_id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.chunk.chunk_id)

        # Re-score merged results by normalizing
        if all_results:
            max_s = max(r.score for r in all_results)
            if max_s > 0:
                for r in all_results:
                    r.score = r.score / max_s

        all_results.sort(key=lambda x: x.score, reverse=True)
        top_results = all_results[:top_k]

        reasoning_steps.append(f"检索到 {len(top_results)} 条高相关证据")

        # ── Step 3: Generate initial answer with chain-of-thought ──
        evidence = evidence_text(top_results)
        prompt = self._build_answer_prompt(question, sub_queries, top_results)
        fallback = self._local_answer(question, top_results)
        answer_text = self.llm.generate(prompt, fallback, task="answer")
        reasoning_steps.append("初始回答已生成")

        # ── Step 4: Self-reflection ──
        max_rounds = getattr(self.config, "max_reflection_rounds", 1) if self.config else 1
        final_answer = answer_text
        for round_i in range(max_rounds):
            reflection = self.llm.reflect(question, final_answer, evidence, task="answer")
            score = reflection.get("score", 60)
            reasoning_steps.append(f"反思轮次 {round_i+1}：质量评分 {score}/100")

            if score >= 85:
                reasoning_steps.append("质量达标，无需进一步修改")
                break

            revised = reflection.get("revised_answer")
            if revised and revised != final_answer:
                final_answer = revised
                reasoning_steps.append("已采纳反思建议，生成改进版本")
            else:
                # Apply improvements manually if LLM didn't produce revised_answer
                weaknesses = reflection.get("weaknesses", [])
                improvements = reflection.get("improvements", [])
                if improvements:
                    final_answer = self._apply_improvements(final_answer, improvements)
                    reasoning_steps.append(f"应用了 {len(improvements)} 条改进建议")

        # ── Step 5: Confidence assessment ──
        confidence = self.llm.assess_confidence(question, final_answer, evidence)
        reasoning_steps.append(f"最终置信度：{confidence:.0%}")

        logger.info(
            f"Answered question in {time.perf_counter()-start:.2f}s; "
            f"evidence={len(top_results)}, confidence={confidence:.2f}"
        )
        return AgentResponse(
            "检索问答结果",
            final_answer,
            top_results,
            confidence=confidence,
            reasoning_trace="\n".join(reasoning_steps),
        )

    def analyze_paper(self, keyword: str | None = None, file_path: Path | None = None,
                      title_query: str | None = None,
                      save_memory: bool = False) -> AgentResponse:
        """Deep paper analysis with multi-angle evidence gathering."""
        start = time.perf_counter()
        reasoning_steps = []
        retriever = self._retriever()

        # ── Resolve paper title ──
        resolved_title = self._resolve_paper_title(retriever, keyword, file_path, title_query)
        if not resolved_title:
            msg = (f"未找到与指定条件匹配的论文。"
                   f"（keyword={keyword}, file={file_path}, title={title_query}）")
            return AgentResponse("单篇论文分析", msg)

        reasoning_steps.append(f"定位论文：{resolved_title}")

        # ── Multi-angle evidence gathering ──
        angles = {
            "核心方法": "method approach algorithm technique framework model",
            "实验与结果": "experiment evaluation result finding benchmark performance",
            "贡献与创新": "contribution novelty improvement advantage advance",
            "局限与不足": "limitation weakness shortcoming challenge drawback gap",
            "相关工作": "related work baseline comparison prior existing previous",
        }

        all_results: list[SearchResult] = []
        seen_ids: set[str] = set()
        for angle_name, keywords in angles.items():
            q = f"{resolved_title} {keywords}"
            results = retriever.search(q, top_k=4, paper_filter=[resolved_title])
            for r in results:
                if r.chunk.chunk_id not in seen_ids:
                    all_results.append(r)
                    seen_ids.add(r.chunk.chunk_id)
            reasoning_steps.append(f"{angle_name}：检索到 {len(results)} 条证据")

        # ── Generate deep analysis ──
        evidence = evidence_text(all_results)
        prompt = self._build_analysis_prompt(resolved_title, all_results)
        fallback = self._local_analysis_enriched(resolved_title, all_results)
        content = self.llm.generate(prompt, fallback, task="analyze")
        reasoning_steps.append("深度分析已生成")

        # ── Confidence ──
        confidence = self.llm.assess_confidence(
            f"分析论文《{resolved_title}》", content, evidence
        )

        resp = AgentResponse("单篇论文分析", content, all_results,
                             confidence=confidence,
                             reasoning_trace="\n".join(reasoning_steps))

        try:
            if save_memory and self.memory:
                self.memory.add("analysis", resp.content, {"paper": resolved_title})
        except Exception:
            pass

        logger.info(f"Analyzed paper '{resolved_title}' in {time.perf_counter()-start:.2f}s")
        return resp

    def compare_papers(self, keywords: list[str] | None = None,
                       file_paths: list[Path] | None = None) -> AgentResponse:
        """Multi-paper comparison with structured analytical framework."""
        start = time.perf_counter()
        reasoning_steps = []
        retriever = self._retriever()
        matched_titles: list[str] = []

        # Resolve titles
        if file_paths:
            for fp in file_paths:
                found = self._find_title_by_path(retriever, fp)
                if found and found not in matched_titles:
                    matched_titles.append(found)

        if keywords:
            papers = retriever.find_papers([k for k in keywords if k])
            for t in papers:
                if t not in matched_titles:
                    matched_titles.append(t)

        if len(matched_titles) < 2:
            return AgentResponse("多篇论文比较", "匹配到的论文少于 2 篇，请提供更多论文关键词或文件路径。")

        reasoning_steps.append(f"待比较论文：{', '.join(matched_titles)}")

        # Gather evidence per paper
        all_results: list[SearchResult] = []
        for title in matched_titles:
            results = retriever.search(
                f"{title} 研究主题 方法 数据 实验 结果 局限 贡献",
                top_k=6,
                paper_filter=[title],
            )
            all_results.extend(results)
            reasoning_steps.append(f"《{title}》：{len(results)} 条证据")

        evidence = evidence_text(all_results, limit=900)
        prompt = self._build_comparison_prompt(matched_titles, all_results)
        fallback = self._local_comparison(matched_titles, all_results)
        content = self.llm.generate(prompt, fallback, task="compare")
        reasoning_steps.append("比较分析已生成")

        confidence = self.llm.assess_confidence(
            f"比较论文：{', '.join(matched_titles)}", content, evidence
        )

        resp = AgentResponse("多篇论文比较", content, all_results,
                             confidence=confidence,
                             reasoning_trace="\n".join(reasoning_steps))
        logger.info(f"Compared papers ({len(matched_titles)}) in {time.perf_counter()-start:.2f}s")
        return resp

    def generate_outline(self, topic: str) -> AgentResponse:
        """Generate a literature review outline with evidence mapping."""
        start = time.perf_counter()
        reasoning_steps = []

        retriever = self._retriever()

        # Broad search for comprehensive coverage
        results = retriever.search(
            topic + " 综述 研究现状 方法 应用 挑战 未来方向",
            top_k=12,
        )
        reasoning_steps.append(f"综述提纲检索到 {len(results)} 条证据")

        evidence = evidence_text(results, limit=900)
        prompt = self._build_outline_prompt(topic, results)
        fallback = self._local_outline(topic, results)
        content = self.llm.generate(prompt, fallback, task="outline")
        reasoning_steps.append("综述提纲已生成")

        # Write outline and per-section files
        folder_name = safe_filename(topic).replace(".md", "")
        out_dir = self.config.outputs_dir / folder_name
        ensure_dir(out_dir)

        index_path = out_dir / "00_index.md"
        index_path.write_text(f"# 综述提纲：{topic}\n\n" + content, encoding="utf-8")

        sections = self._parse_sections(content)
        if not sections:
            sections = [
                ("引言", ["写作要点：...", "引用线索：请参考索引中的证据。"]),
                ("方法比较", ["写作要点：...", "引用线索：请参考索引中的证据。"]),
                ("未来方向", ["写作要点：...", "引用线索：请参考索引中的证据。"]),
            ]

        file_list = []
        for idx, (title, block) in enumerate(sections, start=1):
            safe = f"{idx:02d}_{safe_filename(title).replace('.md', '')}"
            path = out_dir / f"{safe}.md"
            body = f"# {title}\n\n" + "\n".join(block).strip() + "\n"
            path.write_text(body, encoding="utf-8")
            file_list.append(path)

        summary = f"综述提纲已写入：{index_path}，分节文件：{', '.join(str(p) for p in file_list)}"
        reasoning_steps.append(summary)

        logger.info(f"Generated outline for '{topic}' in {time.perf_counter()-start:.2f}s")
        return AgentResponse("综述提纲", content + "\n\n" + summary, results,
                             reasoning_trace="\n".join(reasoning_steps))

    def run_workflow(self, topic: str, top_k: int = 6) -> Path:
        """Full research assistant workflow with comprehensive analysis."""
        start = time.perf_counter()
        ensure_dir(self.config.outputs_dir)
        retriever = self._retriever()
        papers = sorted({chunk.title for chunk in retriever.chunks})

        qa = self.answer(topic, top_k=top_k)
        analyses = [self.analyze_paper(title) for title in papers[:min(3, len(papers))]]
        comparison = (
            self.compare_papers(papers[:min(3, len(papers))])
            if len(papers) >= 2
            else AgentResponse("多篇论文比较", "本地知识库论文少于 2 篇，跳过比较。")
        )
        outline = self.generate_outline(topic)

        sections = [
            f"# CityScholar-Agent 工作流报告\n\n**研究主题：** {topic}",
            qa.to_markdown(),
            *[item.to_markdown() for item in analyses],
            comparison.to_markdown(),
            outline.to_markdown(),
        ]
        report = "\n\n---\n\n".join(sections)
        path = self.config.outputs_dir / safe_filename(topic)
        path.write_text(report, encoding="utf-8")
        logger.info(f"Workflow finished in {time.perf_counter()-start:.1f}s; report={path}")
        return path

    # ════════════════════════════════════════════════════════════════════════════
    # Query Decomposition
    # ════════════════════════════════════════════════════════════════════════════

    def _decompose_query(self, query: str) -> list[str]:
        """Decompose a complex query into sub-queries for comprehensive retrieval.

        For simple queries (fewer than 8 tokens), returns the original query.
        For complex queries, uses LLM to decompose.
        """
        tokens = query.split()
        if len(tokens) <= 8:
            return [query]

        decomposition = self.llm.generate_json(
            f"将以下学术问题分解为 2-4 个独立的子查询：\n\n{query}",
            {"sub_queries": [query], "analysis": "query too short to decompose"},
            "decompose",
            max_tokens=400,
        )
        sub_queries = decomposition.get("sub_queries", [query])
        analysis = decomposition.get("analysis", "")
        if analysis:
            logger.info(f"Decomposition reasoning: {analysis[:200]}")

        # Ensure original query is always included
        if query not in sub_queries:
            sub_queries.insert(0, query)

        return sub_queries[:4]  # cap at 4 sub-queries

    # ════════════════════════════════════════════════════════════════════════════
    # Prompt Builders
    # ════════════════════════════════════════════════════════════════════════════

    def _build_answer_prompt(self, question: str, sub_queries: list[str],
                             results: list[SearchResult]) -> str:
        parts = [
            f"## 用户问题\n{question}\n",
        ]
        if len(sub_queries) > 1:
            parts.append(f"## 检索策略（分解为 {len(sub_queries)} 个子查询）")
            for i, sq in enumerate(sub_queries, 1):
                parts.append(f"  {i}. {sq}")
            parts.append("")

        parts.extend([
            "## 检索证据",
            evidence_text(results),
            "",
            "## 回答要求",
            "1. 先用 1-2 句话给出核心结论",
            "2. 分点展开论证，每条结论必须标注来源论文 [编号]",
            "3. 如果证据之间存在矛盾或不一致，明确指出",
            "4. 标注哪些是直接证据支持的结论，哪些是基于证据的合理推断",
            "5. 在回答末尾，列出本回答无法覆盖的知识盲区",
        ])
        return "\n".join(parts)

    def _build_analysis_prompt(self, title: str, results: list[SearchResult]) -> str:
        evidence = evidence_text(results)
        return (
            f"## 分析目标\n对论文《{title}》进行深度学术分析。\n\n"
            "## 分析框架\n"
            "请按照以下维度逐一分析，每个维度给出具体、有据可查的评价：\n\n"
            "### 1. 核心创新与贡献\n"
            "- 该论文最独特的贡献是什么？与已有工作相比有何实质性进步？\n"
            "- 这些贡献的理论意义和实际价值分别如何？\n\n"
            "### 2. 方法论深度评价\n"
            "- 技术路线是否合理？关键假设是否过强？\n"
            "- 是否存在可替代的方法或潜在的改进空间？\n\n"
            "### 3. 实验设计与证据质量\n"
            "- 实验规模、数据集选择和评估指标是否充分？\n"
            "- 主要结论是否有充分的实验证据支撑？\n\n"
            "### 4. 局限性与批判性分析\n"
            "- 方法的主要局限是什么？\n"
            "- 哪些结论的推广可能受到限制？\n\n"
            "### 5. 综述写作指导\n"
            "- 该论文适合放入综述的哪个章节？\n"
            "- 推荐引用哪些关键结论？如何组织引用？\n\n"
            "## 证据\n"
            f"{evidence}\n\n"
            "## 输出要求\n"
            "学术论文风格，中文为主，关键术语保留英文原文。标注小标题便于直接复用到 Markdown 文档。"
        )

    def _build_comparison_prompt(self, titles: list[str],
                                results: list[SearchResult]) -> str:
        evidence = evidence_text(results, limit=900)
        title_list = "、".join(f"《{t}》" for t in titles)
        return (
            f"## 比较目标\n对以下论文进行系统性对比分析：{title_list}\n\n"
            "## 比较框架\n"
            "### 1. 研究定位\n"
            "- 各论文在研究版图中的位置：是互补、竞争还是递进关系？\n\n"
            "### 2. 方法对比\n"
            "| 维度 | " + " | ".join(titles[:4]) + " |\n"
            "| --- | " + " | ".join(["---"] * min(len(titles), 4)) + " |\n"
            "| 核心方法 | | |\n"
            "| 数据要求 | | |\n"
            "| 计算复杂度 | | |\n\n"
            "### 3. 证据强度\n"
            "- 各论文的实验规模、数据质量和评估指标的可信度对比\n\n"
            "### 4. 综合洞察\n"
            "- 整合所有论文的发现，提炼任何单一论文未覆盖的全局视角\n\n"
            "### 5. 综述组织建议\n"
            "- 如何在文献综述中安排这些论文的讨论顺序和逻辑\n\n"
            "## 证据\n"
            f"{evidence}\n\n"
            "## 输出要求\n"
            "表格化比较要点 + 文字综合评述。中文为主，关键术语保留英文。"
        )

    def _build_outline_prompt(self, topic: str, results: list[SearchResult]) -> str:
        evidence = evidence_text(results, limit=900)
        return (
            f"## 综述主题\n{topic}\n\n"
            "## 请生成\n\n"
            "### 1. 候选综述题目（3 个，不同侧重角度）\n"
            "- 题目 A（侧重技术方法）\n"
            "- 题目 B（侧重应用场景）\n"
            "- 题目 C（侧重发展趋势）\n\n"
            "### 2. 详细综述提纲\n"
            "请按照以下标准结构生成（可根据实际论文调整）：\n"
            "- # 引言\n  - ## 研究背景\n  - ## 综述范围与方法\n"
            "- # 理论基础与技术框架\n  - ## 核心概念\n  - ## 关键技术\n"
            "- # 方法分类与比较\n  - ## 方法一（按论文归纳）\n  - ## 方法二...\n"
            "- # 应用领域\n  - ## 场景一\n  - ## 场景二\n"
            "- # 现有挑战与局限\n"
            "- # 未来研究方向\n"
            "- # 结论\n\n"
            "每个一级标题需要包含：\n"
            "1. 写作提示（3-6 句说明该节应论证的核心论点）\n"
            "2. 可直接引用的论文线索（给出论文标题或短引用）\n"
            "3. 识别的研究空白（现有论文未充分覆盖的领域）\n\n"
            "## 证据\n"
            f"{evidence}\n\n"
            "## 输出格式\n"
            "Markdown，一级标题用 #，二级标题用 ##，写作指导紧跟标题。"
        )

    # ════════════════════════════════════════════════════════════════════════════
    # Reflection Helpers
    # ════════════════════════════════════════════════════════════════════════════

    def _apply_improvements(self, answer: str, improvements: list[str]) -> str:
        """Apply reflection improvements to the answer."""
        if not improvements:
            return answer
        addition = "\n\n**补充说明：**\n"
        for i, imp in enumerate(improvements, 1):
            addition += f"{i}. {imp}\n"
        return answer + addition

    # ════════════════════════════════════════════════════════════════════════════
    # Paper Resolution Helpers
    # ════════════════════════════════════════════════════════════════════════════

    def _resolve_paper_title(self, retriever: HybridRetriever,
                             keyword: str | None = None,
                             file_path: Path | None = None,
                             title_query: str | None = None) -> str | None:
        """Resolve a paper title from various input methods.

        Returns the best match, or None if no match found.
        """
        # 1. File path (most specific)
        if file_path:
            fp = str(Path(file_path).resolve())
            for chunk in retriever.chunks:
                try:
                    if Path(chunk.source_path).resolve() == Path(fp):
                        return chunk.title
                except Exception:
                    if Path(chunk.source_path).name == Path(fp).name:
                        return chunk.title

        # 2. Title query (exact or semantic)
        if title_query:
            papers = retriever.find_papers([title_query])
            if papers:
                # Return the first (best) match
                return next(iter(papers.keys()))
            sr = retriever.search(title_query, top_k=1)
            if sr:
                return sr[0].chunk.title

        # 3. Keyword match
        if keyword:
            papers = retriever.find_papers([keyword])
            if papers:
                return next(iter(papers.keys()))

        return None

    def list_papers(self) -> list[str]:
        """List all paper titles in the knowledge base."""
        retriever = self._retriever()
        return retriever.list_all_papers()

    def find_paper_candidates(self, query: str, max_candidates: int = 10) -> list[str]:
        """Find candidate paper titles matching a query string.

        Returns a list of matching titles ranked by relevance.
        Useful for presenting choices to the user.
        """
        retriever = self._retriever()
        papers = retriever.find_papers([query])
        candidates = list(papers.keys())[:max_candidates]
        return candidates

    def _find_title_by_path(self, retriever: HybridRetriever, fp: Path) -> str | None:
        for chunk in retriever.chunks:
            try:
                if Path(chunk.source_path).resolve() == fp.resolve():
                    return chunk.title
            except Exception:
                if Path(chunk.source_path).name == fp.name:
                    return chunk.title
        return None

    def _parse_sections(self, content: str) -> list[tuple[str, list[str]]]:
        """Parse markdown content into sections based on top-level headings."""
        lines = content.splitlines()
        sections: list[tuple[str, list[str]]] = []
        current_title = None
        current_block: list[str] = []
        for line in lines:
            if line.strip().startswith("# "):
                if current_title:
                    sections.append((current_title, current_block))
                current_title = line.strip().lstrip("# ").strip()
                current_block = []
            else:
                if current_title:
                    current_block.append(line)
        if current_title:
            sections.append((current_title, current_block))
        return sections

    # ════════════════════════════════════════════════════════════════════════════
    # Retriever & Local Fallbacks
    # ════════════════════════════════════════════════════════════════════════════

    def _retriever(self) -> HybridRetriever:
        loaded = self.indexer.load()
        if len(loaded) == 3:
            chunks, vectorizer, matrix = loaded
            embeddings, faiss_index, faiss_id_map = None, None, None
        elif len(loaded) == 4:
            chunks, vectorizer, matrix, embeddings = loaded
            faiss_index, faiss_id_map = None, None
        else:
            chunks, vectorizer, matrix, embeddings, faiss_index, faiss_id_map = loaded
        return HybridRetriever(
            chunks, vectorizer, matrix, embeddings,
            faiss_index, faiss_id_map, config=self.config,
        )

    def _local_answer(self, question: str, results: list[SearchResult]) -> str:
        if not results:
            return "本地知识库中没有检索到足够证据。"
        lines = [f"围绕问题\"{question}\"，本地知识库中最相关的证据显示："]
        for item in results[:4]:
            lines.append(f"- 《{item.chunk.title}》：{item.chunk.text[:180]}...")
        lines.append("以上回答为本地抽取式结果；配置大模型 API 后可生成更连贯的综合回答。")
        return "\n".join(lines)

    def _local_analysis_enriched(self, title: str, results: list[SearchResult]) -> str:
        """Enhanced local fallback for paper analysis."""
        if not results:
            return "本地知识库中没有检索到足够证据。"

        top = results[:8]
        raw_snips = [item.chunk.text.replace("\n", " ") for item in top]

        # Simple term replacement for readability
        repl = {
            "regret": "遗憾/累积遗憾", "differential privacy": "差分隐私",
            "jointly differential privacy": "联合差分隐私", "user-level": "用户级",
            "bandit": "Bandit（臂/带概率决策问题）", "contextual": "上下文的",
            "LinUCB": "LinUCB（线性上置信区间）", "DP-MAB": "差分隐私多臂赌博（DP-MAB）",
            "clients": "客户端", "horizon": "时间跨度 (T)", "algorithm": "算法",
            "proof": "证明", "shuffle": "洗牌 (shuffle)",
        }

        def pseudo_translate(s: str) -> str:
            t = s
            for k, v in repl.items():
                t = t.replace(k, v).replace(k.capitalize(), v)
            return t[:600].strip() if len(t) > 600 else t.strip()

        translated = None
        try:
            translated = self._try_translate_texts(raw_snips)
        except Exception:
            translated = None
        if not translated:
            translated = [pseudo_translate(s) for s in raw_snips]

        evidence_lines = []
        for idx, item in enumerate(top, start=1):
            evidence_lines.append(f"{idx}. {item.chunk.citation()} — {translated[idx-1]}")

        parts = [
            "# 单篇论文分析（本地回退）",
            f"## 论文：《{title}》",
            "### 摘要（合成）",
            f"本段基于论文《{title}》的抽取证据合成：\n{translated[0] if translated else '（无）'}",
            "### 研究问题与背景",
            f"背景线索：{translated[1] if len(translated) > 1 else '无更多线索'}",
            "### 方法与实验设计",
            f"可提取的技术线索：{translated[2] if len(translated) > 2 else '无'}",
            "### 主要结果与贡献",
            f"证据片段：{translated[3] if len(translated) > 3 else '无'}",
            "### 局限性与建议",
            "当前以理论分析为主。建议补充具体噪声机制、参数配置与仿真实验。",
            "### 证据片段（可读化）",
            "\n".join(evidence_lines),
        ]
        return "\n\n".join(parts)

    def _local_comparison(self, titles: list[str], results: list[SearchResult]) -> str:
        lines = ["| 论文 | 可比较维度 | 证据线索 |", "|---|---|---|"]
        for title in titles:
            evidence = next(
                (item.chunk.text[:160] for item in results if item.chunk.title == title),
                "暂无证据",
            )
            lines.append(f"| {title} | 研究目标、方法、数据、结论、局限 | {evidence}... |")
        return "\n".join(lines)

    def _local_outline(self, topic: str, results: list[SearchResult]) -> str:
        citations = "\n".join(
            f"- {item.chunk.citation()}: {item.chunk.text[:150]}..."
            for item in results[:6]
        )
        return (
            f"## 题目：{topic}\n\n"
            "1. 引言：说明研究背景、问题重要性和综述范围。\n"
            "2. 相关技术基础：介绍大语言模型、RAG、Agent 工作流与本地知识库。\n"
            "3. 典型应用场景：按论文主题归纳主要应用。\n"
            "4. 方法比较：比较不同论文的数据、模型、检索方式和评估指标。\n"
            "5. 现有挑战：讨论可靠性、可解释性、成本、隐私和安全问题。\n"
            "6. 未来方向：提出多模态、GraphRAG、多智能体协同和领域评测等方向。\n\n"
            "### 可引用证据\n"
            f"{citations}"
        )

    def _try_translate_texts(self, texts: list[str]) -> list[str] | None:
        """Try to translate English text chunks to Chinese using Helsinki-NLP model."""
        try:
            from transformers import pipeline
            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-zh")
            out = []
            batch_size = 4
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                res = translator(batch)
                for r in res:
                    out.append(r.get("translation_text", "").replace("\n", " ").strip())
            return out
        except Exception:
            return None

    def _local_summarize_texts(self, texts: list[str], n_sentences: int = 4) -> str:
        """Extractive local summary using TF-IDF sentence scoring."""
        import re
        joined = "\n\n".join(t for t in texts if t)
        sents = [s.strip() for s in re.split(r"(?<=[。.!?\n])\s+", joined) if s.strip()]
        if not sents:
            return joined[:600]
        sents = [s if len(s) <= 800 else s[:800] + "..." for s in sents]

        try:
            import numpy as _np
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20000)
                X = vec.fit_transform(sents)
                sims = (X * X.T).toarray()
            except Exception:
                sims = None
            if sims is None:
                sims = _np.zeros((len(sents), len(sents)), dtype=float)
                for i in range(len(sents)):
                    xi = set(re.findall(r"\w+", sents[i].lower()))
                    for j in range(i, len(sents)):
                        xj = set(re.findall(r"\w+", sents[j].lower()))
                        inter = len(xi & xj)
                        denom = max(1, len(xi) + len(xj))
                        sims[i, j] = sims[j, i] = inter / denom

            try:
                import networkx as nx
                G = nx.Graph()
                for i in range(len(sents)):
                    for j in range(i + 1, len(sents)):
                        w = float(sims[i, j])
                        if w > 0.0:
                            G.add_edge(i, j, weight=w)
                pr = nx.pagerank(G, weight="weight") if len(G) > 0 else {i: 1.0 for i in range(len(sents))}
                scores = [pr.get(i, 0.0) for i in range(len(sents))]
            except Exception:
                scores = [float(_np.sum(sims[i])) for i in range(len(sents))]
        except Exception:
            scores = [len(s) for s in sents]

        try:
            import numpy as _np
            topk = min(n_sentences, len(sents))
            idx = list(_np.argsort(-_np.array(scores))[:topk])
            idx_sorted = sorted(idx)
        except Exception:
            idx_sorted = list(range(min(n_sentences, len(sents))))

        summary = "\n".join(sents[i] for i in idx_sorted)
        evid = []
        for i in idx_sorted[:3]:
            snippet = sents[i][:120].replace("\n", " ")
            evid.append(f"- 证据片段: {snippet}...")
        return summary + "\n\n" + ("\n".join(evid) if evid else "")
