from __future__ import annotations

import logging
import requests
import time
import random
import json

from .config import AppConfig
from .models import SearchResult

logger = logging.getLogger("cityscholar.llm")


# ── Task-specific system prompts ──────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "answer": (
        "你是 CityScholar，一位严谨的学术研究助手。你的职责是基于提供的论文证据，给出准确、有深度的回答。\n"
        "核心原则：\n"
        "1. 只使用提供的证据回答，明确标注每条结论的论文来源\n"
        "2. 对证据进行交叉验证：如果多篇论文结论一致则强调共识，不一致则指出分歧\n"
        "3. 区分「直接证据支持」和「合理推断」，对后者明确标注\n"
        "4. 回答结构：先给出核心结论，再展开论证，最后指出知识盲区\n"
        "5. 使用学术化的中文表达，必要时保留英文术语原文"
    ),
    "analyze": (
        "你是 CityScholar，一位深度论文分析专家。你擅长从论文证据中提取研究脉络、方法论和创新点。\n"
        "分析框架：\n"
        "1. **核心创新**：该论文相比已有工作最独特的贡献是什么？\n"
        "2. **方法论**：技术路线是否合理？是否有可替代方案？\n"
        "3. **证据链**：主要结论是否有充分的实验证据支撑？\n"
        "4. **可引用性**：哪些结论可以直接用于综述写作？如何组织引用？\n"
        "5. **批判性评价**：方法假设是否过强？实验规模是否充分？结论推广是否受限？\n"
        "输出要求：学术论文风格，中文为主，关键术语保留英文原文。"
    ),
    "compare": (
        "你是 CityScholar，擅长多论文对比分析与综合评价。\n"
        "比较框架：\n"
        "1. **研究定位**：各论文在研究版图中的位置（互补/竞争/递进关系）\n"
        "2. **方法差异**：核心技术路线、算法选择、实验设计的异同\n"
        "3. **证据强度**：各论文实验规模、数据质量、评估指标的可信度对比\n"
        "4. **综合洞察**：整合所有论文的发现，提炼出任何单一论文未覆盖的全局视角\n"
        "5. **综述建议**：如何在文献综述中组织这些论文的讨论顺序\n"
        "输出：表格化比较要点 + 文字综合评述，便于直接复用到综述草稿。"
    ),
    "outline": (
        "你是 CityScholar，学术综述结构规划专家。\n"
        "规划原则：\n"
        "1. 综述结构应覆盖：引言→背景基础→方法分类→应用领域→挑战与局限→未来方向\n"
        "2. 每个章节都必须关联到具体的论文证据，标注可引用的论文线索\n"
        "3. 识别研究空白：现有论文未充分覆盖的领域\n"
        "4. 提供 3 个候选综述题目（不同侧重角度）\n"
        "5. 为每个一级标题生成 3-6 句写作指导，说明该节应论证的核心论点\n"
        "输出格式：Markdown，一级标题 # ，二级标题 ## ，写作指导紧跟标题。"
    ),
    "decompose": (
        "你是 CityScholar 查询分解助手。将用户的复杂学术问题分解为 2-4 个独立的子查询，"
        "每个子查询应聚焦一个具体的方面，且可以直接用于论文检索。\n"
        "输出 JSON 格式：{\"sub_queries\": [\"子查询1\", \"子查询2\", ...], \"analysis\": \"分解理由\"}"
    ),
    "reflection": (
        "你是 CityScholar 的自我评审专家。你需要客观评估上一步回答的质量，并指出需要改进的地方。\n"
        "评估维度：\n"
        "1. **证据充分性**：回答是否充分利用了可用证据？\n"
        "2. **逻辑严密性**：论证过程是否自洽？有无逻辑跳跃？\n"
        "3. **学术规范性**：引用是否准确？术语使用是否恰当？\n"
        "4. **完整性**：是否遗漏了重要方面？\n"
        "5. **改进建议**：具体列出需要修改或补充的内容。\n"
        "输出 JSON 格式：{\"score\": 0-100, \"strengths\": [...], \"weaknesses\": [...], \"improvements\": [...], \"revised_answer\": \"改进后的完整回答\"}"
    ),
    "confidence": (
        "基于检索证据和回答内容，评估回答的置信度。考虑：证据覆盖度、证据一致性、回答与证据的匹配度。\n"
        "输出 JSON：{\"confidence\": 0.0-1.0, \"reason\": \"评估理由\"}"
    ),
}

# ── Keyword-based prompt routing ──────────────────────────────────────────────

_TASK_KEYWORDS = {
    "analyze": ["分析", "论文分析", "研究问题", "方法", "贡献", "局限", "analyze", "单篇"],
    "compare": ["比较", "对比", "差异", "异同", "compare", "综述对比", "多篇"],
    "outline": ["提纲", "大纲", "综述结构", "outline", "框架", "章节"],
    "answer": [],  # default
}


def _detect_task(prompt: str) -> str:
    """Detect task type from prompt content for system prompt selection."""
    lower = prompt.lower()
    for task, keywords in _TASK_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return task
    return "answer"


# ── LLM Client ────────────────────────────────────────────────────────────────


class LLMClient:
    def __init__(self, config: AppConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(self.config.llm_api_key and self.config.llm_base_url)

    def _call_api(self, messages: list[dict], temperature: float = 0.2,
                  max_tokens: int = 2048) -> str | None:
        """Single API call attempt. Returns content or None on failure."""
        if not self.enabled:
            return None
        url = self.config.llm_base_url.rstrip("/") + "/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        attempts = max(1, int(getattr(self.config, "llm_retries", 2)))
        timeout = int(getattr(self.config, "llm_timeout", 45))
        base = float(getattr(self.config, "llm_backoff_base", 1.0))

        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                logger.info(f"调用模型 (attempt {attempt}/{attempts}): {self.config.llm_model}")
                response = requests.post(url, json=payload, headers=headers, timeout=timeout)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"].strip()
                logger.info(f"大模型调用成功，输出 {len(content)} 字符。")
                return content
            except Exception as error:
                last_exc = error
                logger.warning(f"调用失败({attempt}/{attempts}): {error}")
                if attempt < attempts:
                    sleep_t = base * (2 ** (attempt - 1)) * (0.8 + 0.4 * random.random())
                    logger.info(f"等待 {sleep_t:.1f}s 后重试...")
                    try:
                        time.sleep(sleep_t)
                    except Exception:
                        pass

        logger.error(f"多次调用失败，使用本地回退。最后错误：{last_exc}")
        return None

    def generate(self, prompt: str, fallback: str, task: str | None = None,
                 max_tokens: int = 2048) -> str:
        """Generate a response with task-appropriate system prompt."""
        if not self.enabled:
            logger.info("未检测到 API Key 或接口地址，使用本地生成。")
            return fallback

        task = task or _detect_task(prompt)
        system_msg = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["answer"])

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        result = self._call_api(messages, temperature=0.2, max_tokens=max_tokens)
        return result if result else fallback

    def generate_json(self, prompt: str, fallback: dict, task: str,
                      max_tokens: int = 1024) -> dict:
        """Generate a JSON-structured response. Returns parsed dict or fallback."""
        if not self.enabled:
            return fallback

        system_msg = SYSTEM_PROMPTS.get(task, SYSTEM_PROMPTS["answer"])
        json_instruction = "\n\n**严格要求：你必须且只返回一个合法的 JSON 对象，不要包含任何其他文字、markdown 标记或代码块符号。只返回纯 JSON。**"

        messages = [
            {"role": "system", "content": system_msg + json_instruction},
            {"role": "user", "content": prompt},
        ]
        result = self._call_api(messages, temperature=0.1, max_tokens=max_tokens)
        if not result:
            return fallback

        # Try to parse JSON from response
        try:
            # Strip common non-JSON wrappers
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Strategy 2: find the outermost {...} block with balanced braces
        import re
        try:
            # Find first '{' and match balanced braces
            start = result.find('{')
            if start >= 0:
                depth = 0
                in_string = False
                escape = False
                for i in range(start, len(result)):
                    c = result[i]
                    if escape:
                        escape = False
                        continue
                    if c == '\\' and in_string:
                        escape = True
                        continue
                    if c == '"' and not escape:
                        in_string = not in_string
                    if not in_string:
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                candidate = result[start:i+1]
                                try:
                                    return json.loads(candidate)
                                except json.JSONDecodeError:
                                    # Try fixing common issues: Chinese quotes, trailing commas
                                    fixed = candidate.replace('\u201c', '"').replace('\u201d', '"')
                                    fixed = re.sub(r',\s*([}}\]])', r'\1', fixed)
                                    return json.loads(fixed)
        except Exception:
            pass

        logger.warning("Failed to parse JSON from LLM response, using fallback")
        return fallback

    def reflect(self, prompt: str, previous_answer: str, evidence: str,
                task: str = "answer") -> dict:
        """Run a self-reflection cycle: evaluate the previous answer and suggest improvements.

        Returns dict with keys: score, strengths, weaknesses, improvements, revised_answer
        """
        reflection_prompt = (
            f"## 原始问题\n{prompt}\n\n"
            f"## 可用证据\n{evidence}\n\n"
            f"## 上一步回答\n{previous_answer}\n\n"
            "请评估上述回答的质量，并给出改进版本。"
        )

        default_reflection = {
            "score": 60,
            "strengths": ["回答基本覆盖了问题要点"],
            "weaknesses": ["无法评估"],
            "improvements": [],
            "revised_answer": previous_answer,
        }
        return self.generate_json(reflection_prompt, default_reflection, "reflection")

    def assess_confidence(self, prompt: str, answer: str, evidence: str) -> float:
        """Assess confidence of the answer (0.0 ~ 1.0).

        Uses LLM when available, falls back to a heuristic score based on
        evidence count, answer length, and citation density.
        """
        if not self.enabled:
            return self._heuristic_confidence(answer, evidence)

        confidence_prompt = (
            f"## 问题\n{prompt}\n\n"
            f"## 证据\n{evidence}\n\n"
            f"## 回答\n{answer}\n\n"
            "请评估该回答的置信度，返回 JSON: {\"confidence\": 0.0到1.0的数值, \"reason\": \"理由\"}"
        )
        default = {"confidence": 0.5, "reason": "无法评估"}
        result = self.generate_json(confidence_prompt, default, "confidence", max_tokens=200)
        conf = result.get("confidence", 0.5)
        try:
            return max(0.0, min(1.0, float(conf)))
        except (TypeError, ValueError):
            pass
        # Fallback: try to extract a number from the raw response text
        import re
        try:
            nums = re.findall(r'0?\.\d+|1\.0|\d{1,3}%|\d+/100', str(result))
            for n in nums:
                n = n.strip('%')
                if '/' in n:
                    num, den = n.split('/')
                    val = float(num) / float(den)
                else:
                    val = float(n)
                    if val > 1.0:
                        val = val / 100.0  # handle percentage
                if 0.0 <= val <= 1.0:
                    return val
        except Exception:
            pass
        return self._heuristic_confidence(answer, evidence)

    @staticmethod
    def _heuristic_confidence(answer: str, evidence: str) -> float:
        """Compute a heuristic confidence score when LLM is unavailable or JSON parsing fails."""
        score = 0.5  # baseline
        # More evidence → higher confidence
        evidence_blocks = [b for b in evidence.split('[\n') if b.strip()]
        score += min(0.15, len(evidence_blocks) * 0.03)
        # Longer answer → slightly higher (but cap)
        score += min(0.1, len(answer) / 10000)
        # Check for citation markers like [1], [2], etc.
        import re
        citations = len(re.findall(r'\[\d+\]', answer))
        score += min(0.15, citations * 0.025)
        # Penalize very short answers
        if len(answer) < 200:
            score -= 0.2
        return max(0.1, min(0.9, score))


# ── Evidence formatting ────────────────────────────────────────────────────────

def evidence_text(results: list[SearchResult], limit: int = 900) -> str:
    blocks = []
    for idx, item in enumerate(results, 1):
        text = item.chunk.text.replace("\n", " ")[:limit]
        path_info = ""
        if item.path:
            path_info = f" [via: {' → '.join(item.path[:3])}]"
        blocks.append(f"[{idx}] {item.chunk.citation()}{path_info}\n{text}")
    return "\n\n".join(blocks)
