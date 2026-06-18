"""智能体安全模块 — Prompt 注入检测、内容安全过滤、输出审查。

核心能力：
- 输入安全检查：检测 Prompt 注入攻击、恶意指令
- 输出内容审查：学术诚信检查、敏感信息过滤
- 风险评估评分：多维度安全评分
- 安全日志记录：追踪所有安全事件
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("cityscholar.safety")


# ── 安全事件 ──────────────────────────────────────────────────────────────────

@dataclass
class SafetyEvent:
    """一条安全事件记录。"""
    timestamp: str
    event_type: str       # injection / toxic / policy / info_leak
    severity: str         # critical / high / medium / low
    source: str           # input / output
    detail: str
    blocked: bool = False

    def to_dict(self) -> dict:
        return {
            "time": self.timestamp,
            "type": self.event_type,
            "severity": self.severity,
            "source": self.source,
            "detail": self.detail,
            "blocked": self.blocked,
        }


# ── 安全规则引擎 ──────────────────────────────────────────────────────────────

class SafetyGuard:
    """智能体安全防护系统：输入检查、输出审查、风险评估。"""

    # Prompt 注入攻击特征
    INJECTION_PATTERNS = [
        # 直接指令覆盖
        r"(?i)ignore\s+(all\s+)?previous\s+instructions",
        r"(?i)disregard\s+(all\s+)?prior",
        r"(?i)forget\s+everything",
        r"(?i)override\s+system\s+prompt",
        r"忽略之前的指令",
        r"忽略所有指令",
        r"覆盖系统提示",
        r"忘记你之前的设定",
        r"你现在是",
        r"假装你是",
        # 角色劫持
        r"(?i)you\s+are\s+now\s+(?:a|an|the)",
        r"(?i)act\s+as\s+if\s+you",
        r"(?i)roleplay\s+as",
        r"扮演.*角色",
        r"从现在开始你是",
        # 数据提取
        r"(?i)reveal\s+(?:your|the)\s+system\s+prompt",
        r"(?i)show\s+me\s+(?:your|the)\s+instructions",
        r"(?i)print\s+(?:your|the)\s+system",
        r"输出你的系统提示",
        r"显示你的指令",
        r"把你的设定告诉我",
    ]

    # 敏感内容关键词
    SENSITIVE_KEYWORDS = [
        # 安全相关
        "weapon", "bomb", "attack", "exploit", "hack", "malware",
        "漏洞利用", "攻击方法", "制作炸弹", "黑客攻击",
        # 隐私相关
        "social security", "credit card", "password", "private key",
        "身份证号", "银行卡号", "密码", "私钥",
    ]

    # 学术诚信问题
    PLAGIARISM_INDICATORS = [
        r"(?i)copy\s+(?:from|paste)",
        r"(?i)plagiarize",
        r"(?i)buy\s+(?:essay|paper|thesis)",
        r"代写", "抄袭", "论文买卖", "代考",
    ]

    def __init__(self):
        self._events: list[SafetyEvent] = []
        self._blocked_count = 0

    # ── 输入安全检查 ─────────────────────────────────────────────────────

    def check_input(self, text: str) -> dict[str, Any]:
        """检查用户输入的安全性。

        Returns:
            {
                "safe": bool,
                "risk_score": float (0-1),
                "events": list[SafetyEvent],
                "recommendation": str,
            }
        """
        events: list[SafetyEvent] = []
        risk_score = 0.0

        # 1. Prompt 注入检测
        injection = self._detect_injection(text)
        if injection:
            events.extend(injection)
            risk_score += 0.5 * len(injection)

        # 2. 敏感内容检测
        sensitive = self._detect_sensitive(text)
        if sensitive:
            events.extend(sensitive)
            risk_score += 0.3 * len(sensitive)

        # 3. 学术诚信检测
        plagiarism = self._detect_plagiarism(text)
        if plagiarism:
            events.extend(plagiarism)
            risk_score += 0.2 * len(plagiarism)

        risk_score = min(1.0, risk_score)
        self._events.extend(events)

        safe = risk_score < 0.5
        recommendation = "允许处理"
        if risk_score >= 0.7:
            recommendation = "建议拒绝处理此请求"
        elif risk_score >= 0.5:
            recommendation = "建议人工审核后处理"

        return {
            "safe": safe,
            "risk_score": risk_score,
            "events": events,
            "recommendation": recommendation,
        }

    # ── 输出内容审查 ─────────────────────────────────────────────────────

    def check_output(self, text: str) -> dict[str, Any]:
        """审查 Agent 输出内容的安全性。"""
        events: list[SafetyEvent] = []
        risk_score = 0.0

        # 1. 敏感信息泄露检测
        leaks = self._detect_info_leak(text)
        if leaks:
            events.extend(leaks)
            risk_score += 0.4 * len(leaks)

        # 2. 有害内容检测
        toxic = self._detect_toxic(text)
        if toxic:
            events.extend(toxic)
            risk_score += 0.3 * len(toxic)

        # 3. 幻觉/虚构检测（检查是否编造了不存在的论文引用）
        hallucination = self._detect_hallucination(text)
        if hallucination:
            events.extend(hallucination)
            risk_score += 0.2 * len(hallucination)

        risk_score = min(1.0, risk_score)
        self._events.extend(events)

        return {
            "safe": risk_score < 0.5,
            "risk_score": risk_score,
            "events": events,
        }

    # ── 内部检测方法 ─────────────────────────────────────────────────────

    def _detect_injection(self, text: str) -> list[SafetyEvent]:
        events = []
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text):
                events.append(SafetyEvent(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    event_type="injection",
                    severity="high",
                    source="input",
                    detail=f"检测到 Prompt 注入特征：{pattern[:40]}",
                    blocked=True,
                ))
        return events

    def _detect_sensitive(self, text: str) -> list[SafetyEvent]:
        events = []
        text_lower = text.lower()
        for keyword in self.SENSITIVE_KEYWORDS:
            if keyword.lower() in text_lower:
                events.append(SafetyEvent(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    event_type="policy",
                    severity="medium",
                    source="input",
                    detail=f"检测到敏感关键词：{keyword}",
                ))
        return events

    def _detect_plagiarism(self, text: str) -> list[SafetyEvent]:
        events = []
        for pattern in self.PLAGIARISM_INDICATORS:
            if re.search(pattern, text):
                events.append(SafetyEvent(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    event_type="policy",
                    severity="low",
                    source="input",
                    detail=f"学术诚信提示：检测到相关关键词",
                ))
        return events

    def _detect_info_leak(self, text: str) -> list[SafetyEvent]:
        events = []
        # 检测 API Key 泄露
        if re.search(r'sk-[a-zA-Z0-9]{20,}', text):
            events.append(SafetyEvent(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                event_type="info_leak",
                severity="critical",
                source="output",
                detail="输出中包含疑似 API Key",
                blocked=True,
            ))
        # 检测密码/令牌泄露
        if re.search(r'(?i)(?:password|token|secret)\s*[:=]\s*\S+', text):
            events.append(SafetyEvent(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                event_type="info_leak",
                severity="high",
                source="output",
                detail="输出中包含疑似凭证信息",
                blocked=True,
            ))
        return events

    def _detect_toxic(self, text: str) -> list[SafetyEvent]:
        events = []
        toxic_patterns = [
            r"(?i)you\s+(?:are|should)\s+(?:die|stupid|idiot)",
            r"(?i)I\s+(?:hope|wish)\s+you\s+(?:die|fail)",
        ]
        for pattern in toxic_patterns:
            if re.search(pattern, text):
                events.append(SafetyEvent(
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    event_type="toxic",
                    severity="high",
                    source="output",
                    detail="检测到有害内容",
                    blocked=True,
                ))
        return events

    def _detect_hallucination(self, text: str) -> list[SafetyEvent]:
        """检测可能的学术幻觉（编造引用）。"""
        events = []
        # 检测看起来像编造的论文引用格式
        fake_ref = re.findall(r'\[\d+\]\s+[A-Z][a-z]+\s+et\s+al\.\s*,\s*\d{4}', text)
        if len(fake_ref) > 10:
            events.append(SafetyEvent(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                event_type="policy",
                severity="medium",
                source="output",
                detail=f"检测到 {len(fake_ref)} 条引用，建议验证是否存在虚构引用",
            ))
        return events

    # ── 安全事件查询 ─────────────────────────────────────────────────────

    def get_events(self, limit: int = 50) -> list[dict]:
        return [e.to_dict() for e in self._events[-limit:]]

    def get_stats(self) -> dict:
        from collections import Counter
        type_counts = Counter(e.event_type for e in self._events)
        severity_counts = Counter(e.severity for e in self._events)
        blocked = sum(1 for e in self._events if e.blocked)
        return {
            "total_events": len(self._events),
            "blocked": blocked,
            "by_type": dict(type_counts),
            "by_severity": dict(severity_counts),
        }

    # ── 沙盒执行 ─────────────────────────────────────────────────────────

    def sandbox_execute(self, func, *args, timeout: float = 10.0, **kwargs) -> dict:
        """在受限环境中执行函数，捕获异常和超时。"""
        import threading

        result = {"ok": False, "result": None, "error": None}

        def target():
            try:
                result["result"] = func(*args, **kwargs)
                result["ok"] = True
            except Exception as e:
                result["error"] = str(e)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            result["error"] = f"执行超时（>{timeout}s）"
            self._events.append(SafetyEvent(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                event_type="policy",
                severity="medium",
                source="system",
                detail=f"沙盒执行超时：{func.__name__}",
            ))

        return result
