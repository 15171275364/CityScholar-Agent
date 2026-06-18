"""多智能体协同模块 — 角色分工、任务委派、结果聚合。

实现四种专业角色：
- Reader: 负责论文检索与信息提取
- Analyst: 负责深度分析与批判性评价
- Critic: 负质量评审与改进建议
- Writer: 负责报告撰写与结构化输出

通过 MultiAgentHub 进行任务编排和结果聚合。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("cityscholar.multiagent")


# ── Agent 角色定义 ─────────────────────────────────────────────────────────────

@dataclass
class AgentRole:
    """一个智能体角色的定义。"""
    name: str
    description: str
    call: Callable[..., Any]
    priority: int = 0  # 执行优先级，数值越小越优先


@dataclass
class TaskResult:
    """单个智能体的执行结果。"""
    agent_name: str
    content: str
    metadata: dict = field(default_factory=dict)
    elapsed: float = 0.0
    success: bool = True
    error: str = ""


@dataclass
class WorkflowResult:
    """多智能体协作的完整工作流结果。"""
    task: str
    results: list[TaskResult] = field(default_factory=list)
    final_output: str = ""
    total_elapsed: float = 0.0

    def to_markdown(self) -> str:
        lines = [f"# 多智能体协作报告\n\n**任务：** {self.task}\n"]
        for r in self.results:
            status = "✅" if r.success else "❌"
            lines.append(f"## {status} {r.agent_name}（{r.elapsed:.1f}s）\n")
            if r.error:
                lines.append(f"错误：{r.error}\n")
            else:
                lines.append(r.content + "\n")
        if self.final_output:
            lines.append(f"\n---\n\n## 综合输出\n\n{self.final_output}")
        return "\n".join(lines)


# ── 专业智能体角色 ─────────────────────────────────────────────────────────────

class ReaderAgent:
    """Reader 角色：负责论文检索与信息提取。"""

    def __init__(self, retriever=None):
        self.retriever = retriever

    def __call__(self, task: dict) -> dict:
        query = task.get("query", "")
        top_k = task.get("top_k", 5)
        if not self.retriever:
            return {"error": "Retriever not initialized"}
        results = self.retriever.search(query, top_k=top_k)
        evidence_parts = []
        for idx, r in enumerate(results, 1):
            text = r.chunk.text.replace("\n", " ")[:300]
            evidence_parts.append(f"[{idx}] {r.chunk.citation()} (score={r.score:.3f})\n{text}")
        return {
            "evidence": "\n\n".join(evidence_parts),
            "paper_count": len({r.chunk.title for r in results}),
            "result_count": len(results),
            "results": results,
        }


class AnalystAgent:
    """Analyst 角色：负责深度分析与批判性评价。"""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def __call__(self, task: dict) -> dict:
        evidence = task.get("evidence", "")
        query = task.get("query", "")
        role_prompt = task.get("role_prompt", "")

        prompt = (
            f"作为学术分析专家，请对以下问题进行深度分析：\n\n"
            f"问题：{query}\n\n"
            f"证据：\n{evidence}\n\n"
            f"{role_prompt}\n\n"
            "请从以下维度分析：\n"
            "1. 核心发现与关键论点\n"
            "2. 方法论评价\n"
            "3. 证据质量与充分性\n"
            "4. 知识盲区与局限\n"
        )
        if self.llm and self.llm.enabled:
            content = self.llm.generate(prompt, self._local_fallback(query, evidence), task="analyze")
        else:
            content = self._local_fallback(query, evidence)

        return {"analysis": content}

    @staticmethod
    def _local_fallback(query: str, evidence: str) -> str:
        return (
            f"## 分析：{query}\n\n"
            f"基于检索到的证据进行分析：\n\n{evidence[:800]}\n\n"
            "*以上为本地分析回退，配置大模型 API 后可获得更深入的分析。*"
        )


class CriticAgent:
    """Critic 角色：负责质量评审与改进建议。"""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def __call__(self, task: dict) -> dict:
        content_to_review = task.get("content", "")
        evidence = task.get("evidence", "")

        prompt = (
            "请作为学术质量评审专家，评估以下回答的质量：\n\n"
            f"待评审内容：\n{content_to_review}\n\n"
            f"参考证据：\n{evidence[:500]}\n\n"
            "评估维度：\n"
            "1. 论证逻辑是否严密\n"
            "2. 证据引用是否准确\n"
            "3. 是否存在遗漏或偏见\n"
            "4. 学术规范性\n"
            "5. 综合评分 (1-10) 和改进建议\n"
        )
        if self.llm and self.llm.enabled:
            review = self.llm.generate(prompt, self._local_review(content_to_review), task="answer")
        else:
            review = self._local_review(content_to_review)

        return {"review": review}

    @staticmethod
    def _local_review(content: str) -> str:
        score = min(8, max(3, len(content) // 300))
        return (
            f"## 质量评审\n\n"
            f"- 综合评分：{score}/10\n"
            f"- 内容长度：{len(content)} 字符\n"
            f"- 建议：可进一步补充具体的论文引用证据\n"
        )


class WriterAgent:
    """Writer 角色：负责报告撰写与结构化输出。"""

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def __call__(self, task: dict) -> dict:
        sections = task.get("sections", [])
        topic = task.get("topic", "")

        prompt = (
            f"请基于以下各模块的输出，撰写一份结构化的学术综述报告。\n\n"
            f"主题：{topic}\n\n"
        )
        for sec in sections:
            prompt += f"### {sec.get('title', '未命名')}\n{sec.get('content', '')}\n\n"

        prompt += (
            "要求：\n"
            "1. 结构清晰，使用 Markdown 格式\n"
            "2. 各部分之间有逻辑衔接\n"
            "3. 标注关键证据来源\n"
            "4. 给出综合结论\n"
        )

        if self.llm and self.llm.enabled:
            content = self.llm.generate(prompt, self._local_write(topic, sections), task="outline")
        else:
            content = self._local_write(topic, sections)

        return {"report": content}

    @staticmethod
    def _local_write(topic: str, sections: list[dict]) -> str:
        parts = [f"# {topic}\n"]
        for sec in sections:
            title = sec.get("title", "未命名")
            content = sec.get("content", "")[:500]
            parts.append(f"## {title}\n\n{content}\n")
        parts.append("\n*以上报告为本地生成回退。*")
        return "\n".join(parts)


# ── 多智能体协调中心 ──────────────────────────────────────────────────────────

class MultiAgentHub:
    """多智能体协调中心：管理角色注册、任务分发和结果聚合。"""

    def __init__(self):
        self.agents: dict[str, AgentRole] = {}

    def register(self, name: str, agent: AgentRole) -> None:
        self.agents[name] = agent
        logger.info(f"注册智能体：{name} — {agent.description}")

    def unregister(self, name: str) -> None:
        self.agents.pop(name, None)

    def execute(self, task: dict, participants: list[str] | None = None) -> list[TaskResult]:
        """按优先级顺序执行指定参与者。"""
        targets = participants or sorted(
            self.agents.keys(),
            key=lambda n: self.agents[n].priority,
        )
        results = []
        for name in targets:
            if name not in self.agents:
                continue
            agent = self.agents[name]
            start = time.perf_counter()
            try:
                output = agent.call(task)
                elapsed = time.perf_counter() - start
                results.append(TaskResult(
                    agent_name=name,
                    content=str(output.get("analysis", output.get("review", output.get("report", output.get("evidence", str(output)))))),
                    metadata=output,
                    elapsed=elapsed,
                    success=True,
                ))
                logger.info(f"[{name}] 完成，耗时 {elapsed:.1f}s")
            except Exception as e:
                elapsed = time.perf_counter() - start
                results.append(TaskResult(
                    agent_name=name, content="", elapsed=elapsed,
                    success=False, error=str(e),
                ))
                logger.error(f"[{name}] 失败：{e}")
        return results

    def run_analysis_workflow(self, query: str, retriever=None, llm_client=None,
                              top_k: int = 5) -> WorkflowResult:
        """执行完整的多智能体分析工作流：Reader → Analyst → Critic → Writer。"""
        start = time.perf_counter()
        workflow_result = WorkflowResult(task=query)

        # Step 1: Reader 检索证据
        reader = ReaderAgent(retriever=retriever)
        self.register("reader", AgentRole("reader", "论文检索与信息提取", reader, priority=0))
        read_results = self.execute({"query": query, "top_k": top_k}, ["reader"])

        evidence = ""
        if read_results and read_results[0].success:
            evidence = read_results[0].metadata.get("evidence", "")
        workflow_result.results.extend(read_results)

        # Step 2: Analyst 深度分析
        analyst = AnalystAgent(llm_client=llm_client)
        self.register("analyst", AgentRole("analyst", "深度分析与批判性评价", analyst, priority=1))
        analysis_results = self.execute({
            "query": query, "evidence": evidence,
            "role_prompt": "请特别关注方法论创新和实验设计的严谨性。",
        }, ["analyst"])
        workflow_result.results.extend(analysis_results)

        analysis_content = ""
        if analysis_results and analysis_results[0].success:
            analysis_content = analysis_results[0].metadata.get("analysis", "")

        # Step 3: Critic 质量评审
        critic = CriticAgent(llm_client=llm_client)
        self.register("critic", AgentRole("critic", "质量评审与改进建议", critic, priority=2))
        review_results = self.execute({
            "content": analysis_content, "evidence": evidence,
        }, ["critic"])
        workflow_result.results.extend(review_results)

        # Step 4: Writer 撰写报告
        writer = WriterAgent(llm_client=llm_client)
        self.register("writer", AgentRole("writer", "报告撰写与结构化输出", writer, priority=3))
        sections = [
            {"title": "检索证据", "content": evidence[:1000]},
            {"title": "深度分析", "content": analysis_content[:1000]},
        ]
        write_results = self.execute({
            "topic": query, "sections": sections,
        }, ["writer"])
        workflow_result.results.extend(write_results)

        if write_results and write_results[0].success:
            workflow_result.final_output = write_results[0].metadata.get("report", "")

        workflow_result.total_elapsed = time.perf_counter() - start
        logger.info(f"多智能体工作流完成，总耗时 {workflow_result.total_elapsed:.1f}s")
        return workflow_result
