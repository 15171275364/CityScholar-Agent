"""MCP (Model Context Protocol) 通讯协议模块 — 工具注册、发现、调用。

实现类 MCP 的工具协议：
- ToolRegistry: 工具注册中心，管理所有可用工具
- MCPNode: 节点间消息传递
- Tool Calling: 结构化工具调用接口
- 内置工具集：search、analyze、compare、outline
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("cityscholar.mcp")


# ── 工具定义 ──────────────────────────────────────────────────────────────────

@dataclass
class ToolDef:
    """一个工具的完整定义。"""
    name: str
    description: str
    parameters: dict[str, dict]  # JSON Schema 风格参数定义
    handler: Callable[..., Any]
    tags: list[str] = field(default_factory=list)

    def to_schema(self) -> dict:
        """输出为 JSON Schema 格式（供 LLM 理解）。"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": [
                    k for k, v in self.parameters.items()
                    if v.get("required", False)
                ],
            },
        }


@dataclass
class ToolCall:
    """一次工具调用请求。"""
    tool_name: str
    arguments: dict[str, Any]
    call_id: str = ""

    def to_dict(self) -> dict:
        return {"tool": self.tool_name, "arguments": self.arguments, "call_id": self.call_id}


@dataclass
class ToolResult:
    """一次工具调用的结果。"""
    call_id: str
    tool_name: str
    success: bool
    output: Any = None
    error: str = ""
    elapsed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "tool": self.tool_name,
            "success": self.success,
            "output": str(self.output)[:500] if self.output else None,
            "error": self.error,
            "elapsed": round(self.elapsed, 3),
        }


# ── 工具注册中心 ──────────────────────────────────────────────────────────────

class ToolRegistry:
    """工具注册中心：管理所有可用工具的注册、发现和调用。"""

    def __init__(self):
        self._tools: dict[str, ToolDef] = {}
        self._call_counter = 0
        self._call_log: list[ToolResult] = []

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool
        logger.info(f"注册工具：{tool.name} — {tool.description[:60]}")

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        """返回所有工具的 Schema 描述（供 LLM 使用）。"""
        return [t.to_schema() for t in self._tools.values()]

    def list_by_tag(self, tag: str) -> list[ToolDef]:
        return [t for t in self._tools.values() if tag in t.tags]

    def call(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """执行一次工具调用。"""
        self._call_counter += 1
        call_id = f"call_{self._call_counter}"

        tool = self._tools.get(tool_name)
        if not tool:
            result = ToolResult(call_id, tool_name, False, error=f"工具不存在：{tool_name}")
            self._call_log.append(result)
            return result

        start = time.perf_counter()
        try:
            output = tool.handler(**arguments)
            elapsed = time.perf_counter() - start
            result = ToolResult(call_id, tool_name, True, output=output, elapsed=elapsed)
            logger.info(f"工具调用成功：{tool_name}（{elapsed:.2f}s）")
        except Exception as e:
            elapsed = time.perf_counter() - start
            result = ToolResult(call_id, tool_name, False, error=str(e), elapsed=elapsed)
            logger.error(f"工具调用失败：{tool_name} — {e}")

        self._call_log.append(result)
        return result

    def call_batch(self, calls: list[ToolCall]) -> list[ToolResult]:
        """批量执行工具调用。"""
        return [self.call(c.tool_name, c.arguments) for c in calls]

    def get_call_log(self) -> list[dict]:
        return [r.to_dict() for r in self._call_log]

    def generate_tools_prompt(self) -> str:
        """生成供 LLM 使用的工具描述文本。"""
        if not self._tools:
            return "当前没有可用的工具。"
        parts = ["## 可用工具\n"]
        for tool in self._tools.values():
            schema = tool.to_schema()
            params_desc = ", ".join(
                f"{k}({v.get('type', 'any')})" +
                (" [必填]" if v.get("required") else "")
                for k, v in schema["parameters"]["properties"].items()
            )
            parts.append(f"- **{tool.name}**: {tool.description}\n  参数：{params_desc or '无'}")
        return "\n".join(parts)


# ── MCP 节点 ──────────────────────────────────────────────────────────────────

class MCPNode:
    """MCP 节点：支持消息收发和工具调用的通讯实体。"""

    def __init__(self, node_id: str, registry: ToolRegistry | None = None):
        self.node_id = node_id
        self.registry = registry or ToolRegistry()
        self.handlers: dict[str, Callable] = {}
        self._message_log: list[dict] = []

    def send(self, target: MCPNode, msg: dict) -> dict:
        """向目标节点发送消息。"""
        msg["_source"] = self.node_id
        msg["_timestamp"] = time.time()
        self._message_log.append(msg)

        msg_type = msg.get("type", "")
        if msg_type == "tool_call":
            tool_name = msg.get("tool", "")
            args = msg.get("arguments", {})
            result = target.registry.call(tool_name, args)
            return result.to_dict()
        elif msg_type in target.handlers:
            return target.handlers[msg_type](msg)
        return {"error": f"no handler for type '{msg_type}'", "target": target.node_id}

    def register_handler(self, msg_type: str, handler: Callable) -> None:
        self.handlers[msg_type] = handler

    def handle(self, msg: dict) -> dict:
        msg_type = msg.get("type", "")
        if msg_type in self.handlers:
            return self.handlers[msg_type](msg)
        return {"error": f"no handler for '{msg_type}'", "node": self.node_id}


# ── 内置工具集 ────────────────────────────────────────────────────────────────

def register_builtins(registry: ToolRegistry, agent=None) -> None:
    """注册 CityScholar 的内置工具集。"""
    if agent is None:
        return

    # 工具 1: 搜索论文
    def search_papers(query: str, top_k: int = 5) -> str:
        retriever = agent._retriever()
        results = retriever.search(query, top_k=top_k)
        if not results:
            return "未找到相关论文。"
        parts = []
        for idx, r in enumerate(results, 1):
            text = r.chunk.text.replace("\n", " ")[:200]
            parts.append(f"[{idx}] {r.chunk.citation()} (score={r.score:.3f})\n{text}")
        return "\n\n".join(parts)

    registry.register(ToolDef(
        name="search_papers",
        description="在本地论文知识库中检索相关内容",
        parameters={
            "query": {"type": "string", "description": "检索关键词或问题", "required": True},
            "top_k": {"type": "integer", "description": "返回结果数量", "default": 5},
        },
        handler=search_papers,
        tags=["retrieval", "search"],
    ))

    # 工具 2: 分析论文
    def analyze_paper(keyword: str, save_memory: bool = False) -> str:
        result = agent.analyze_paper(keyword=keyword, save_memory=save_memory)
        return result.content

    registry.register(ToolDef(
        name="analyze_paper",
        description="对指定论文进行深度学术分析",
        parameters={
            "keyword": {"type": "string", "description": "论文标题关键词", "required": True},
            "save_memory": {"type": "boolean", "description": "是否保存到记忆库", "default": False},
        },
        handler=analyze_paper,
        tags=["analysis", "paper"],
    ))

    # 工具 3: 比较论文
    def compare_papers(keywords: str) -> str:
        kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
        result = agent.compare_papers(keywords=kw_list)
        return result.content

    registry.register(ToolDef(
        name="compare_papers",
        description="比较多篇论文的异同",
        parameters={
            "keywords": {"type": "string", "description": "论文关键词，逗号分隔（至少2个）", "required": True},
        },
        handler=compare_papers,
        tags=["comparison", "paper"],
    ))

    # 工具 4: 生成综述提纲
    def generate_outline(topic: str) -> str:
        result = agent.generate_outline(topic)
        return result.content

    registry.register(ToolDef(
        name="generate_outline",
        description="根据本地论文生成文献综述提纲",
        parameters={
            "topic": {"type": "string", "description": "综述主题", "required": True},
        },
        handler=generate_outline,
        tags=["outline", "writing"],
    ))

    # 工具 5: 列出论文
    def list_papers() -> str:
        papers = agent.list_papers()
        if not papers:
            return "知识库中没有论文。"
        return "\n".join(f"[{i+1}] {t}" for i, t in enumerate(papers))

    registry.register(ToolDef(
        name="list_papers",
        description="列出知识库中所有论文的标题",
        parameters={},
        handler=list_papers,
        tags=["list", "metadata"],
    ))

    # 工具 6: 问答
    def ask_question(question: str, top_k: int = 5) -> str:
        result = agent.answer(question, top_k=top_k)
        return result.content

    registry.register(ToolDef(
        name="ask_question",
        description="基于论文知识库回答学术问题",
        parameters={
            "question": {"type": "string", "description": "学术问题", "required": True},
            "top_k": {"type": "integer", "description": "检索证据数量", "default": 5},
        },
        handler=ask_question,
        tags=["qa", "retrieval"],
    ))

    logger.info(f"已注册 {len(registry._tools)} 个内置工具")
