"""多模态处理模块 — PDF 图表提取、图像文字识别、多模态 Prompt 构造。

核心能力：
- 从 PDF 中提取嵌入的图表（图片流解析）
- 图像 OCR 文字识别（Tesseract / 纯文本回退）
- 图表类型识别（表格 / 流程图 / 折线图等启发式判断）
- 多模态 LLM Prompt 构造（支持视觉模型调用）
"""

from __future__ import annotations

import io
import re
import struct
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger("cityscholar.multimodal")


@dataclass
class ExtractedImage:
    """从 PDF 中提取的图片元信息。"""
    page: int
    width: int = 0
    height: int = 0
    data: bytes = field(default=b"")
    source_path: str = ""
    ocr_text: str = ""
    chart_type: str = "unknown"  # table, chart, diagram, figure, unknown

    @property
    def size_kb(self) -> float:
        return len(self.data) / 1024


class MultiModalProcessor:
    """多模态处理器：从学术论文 PDF 中提取图表并进行分析。"""

    # ── PDF 图表提取 ───────────────────────────────────────────────────────

    def extract_images_from_pdf(self, pdf_path: Path, max_images: int = 20) -> list[ExtractedImage]:
        """从 PDF 文件中提取嵌入的图片。

        优先使用 pypdf 解析图片流；回退到原始字节流解析。
        """
        images: list[ExtractedImage] = []
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(pdf_path))
            for page_idx, page in enumerate(reader.pages, 1):
                if len(images) >= max_images:
                    break
                try:
                    if "/XObject" in (page.get("/Resources") or {}):
                        x_objects = page["/Resources"]["/XObject"].get_object()
                        for obj_name in x_objects:
                            obj = x_objects[obj_name].get_object()
                            if obj.get("/Subtype") == "/Image":
                                img_data = self._decode_image(obj)
                                if img_data and len(img_data) > 500:
                                    images.append(ExtractedImage(
                                        page=page_idx,
                                        width=int(obj.get("/Width", 0)),
                                        height=int(obj.get("/Height", 0)),
                                        data=img_data,
                                        source_path=str(pdf_path),
                                    ))
                except Exception:
                    continue
        except ImportError:
            images = self._extract_images_fallback(pdf_path, max_images)
        except Exception as e:
            logger.warning(f"PDF 图片提取失败: {e}")
            images = self._extract_images_fallback(pdf_path, max_images)

        # 对每张图片做 OCR 和类型识别
        for img in images:
            img.ocr_text = self._ocr_image(img.data)
            img.chart_type = self._classify_chart(img)

        logger.info(f"从 {pdf_path.name} 提取了 {len(images)} 张图片")
        return images

    def _decode_image(self, pdf_obj: Any) -> bytes:
        """从 PDF 图片对象解码像素数据。"""
        try:
            filter_type = pdf_obj.get("/Filter", "")
            data = pdf_obj.get_data()
            if filter_type == "/DCTDecode":
                return data  # JPEG
            if filter_type == "/FlateDecode":
                try:
                    return zlib.decompress(data)
                except Exception:
                    return data
            if filter_type == "/JPXDecode":
                return data  # JPEG2000
            return data
        except Exception:
            return b""

    def _extract_images_fallback(self, pdf_path: Path, max_images: int) -> list[ExtractedImage]:
        """回退方案：从 PDF 字节流中扫描 JPEG/PNG 图片。"""
        images = []
        try:
            raw = pdf_path.read_bytes()
        except Exception:
            return images

        # 扫描 JPEG
        for match in re.finditer(rb'\xff\xd8\xff', raw):
            start = match.start()
            end = raw.find(rb'\xff\xd9', start)
            if end > start and end - start < 10_000_000:
                img_data = raw[start:end + 2]
                if len(img_data) > 1000:
                    images.append(ExtractedImage(page=0, data=img_data, source_path=str(pdf_path)))
                    if len(images) >= max_images:
                        break

        # 扫描 PNG
        for match in re.finditer(rb'\x89PNG\r\n\x1a\n', raw):
            start = match.start()
            # 查找 IEND
            end = raw.find(b'IEND', start)
            if end > start:
                img_data = raw[start:end + 8]
                if len(img_data) > 1000:
                    images.append(ExtractedImage(page=0, data=img_data, source_path=str(pdf_path)))
                    if len(images) >= max_images:
                        break

        return images

    # ── OCR 文字识别 ─────────────────────────────────────────────────────

    def _ocr_image(self, img_data: bytes) -> str:
        """对图片执行 OCR 文字识别。"""
        if not img_data:
            return ""
        # 方案 1: Tesseract
        try:
            from PIL import Image
            import pytesseract
            img = Image.open(io.BytesIO(img_data))
            text = pytesseract.image_to_string(img, lang="eng+chi_sim")
            if text.strip():
                return text.strip().replace("\n", " ")
        except Exception:
            pass

        # 方案 2: 如果是纯文本图片（如表格截图），尝试简单解析
        try:
            from PIL import Image
            img = Image.open(io.BytesIO(img_data))
            # 检查是否是灰度/二值图（可能是表格截图）
            if img.mode in ("L", "1"):
                return f"[灰度图像 {img.size[0]}x{img.size[1]}，建议使用 Tesseract OCR]"
        except Exception:
            pass

        return ""

    # ── 图表类型识别 ─────────────────────────────────────────────────────

    def _classify_chart(self, img: ExtractedImage) -> str:
        """启发式判断图表类型。"""
        text = img.ocr_text.lower()

        # 关键词启发式
        table_kw = ["table", "表格", "row", "column", "cell", "data", "result", "comparison"]
        chart_kw = ["figure", "fig", "chart", "plot", "accuracy", "loss", "epoch", "curve"]
        diagram_kw = ["framework", "architecture", "pipeline", "workflow", "method", "model", "overview"]

        scores = {
            "table": sum(1 for kw in table_kw if kw in text),
            "chart": sum(1 for kw in chart_kw if kw in text),
            "diagram": sum(1 for kw in diagram_kw if kw in text),
        }

        if max(scores.values()) >= 2:
            return max(scores, key=scores.get)

        # 宽高比启发式
        if img.width > 0 and img.height > 0:
            ratio = img.width / img.height
            if ratio > 2.5:
                return "chart"   # 宽扁 → 可能是折线图/柱状图
            if ratio < 0.6:
                return "table"  # 窄长 → 可能是表格

        return "figure"

    # ── 多模态 Prompt 构造 ──────────────────────────────────────────────

    def build_multimodal_prompt(self, text: str, images: list[ExtractedImage] | None = None,
                                 task: str = "analysis") -> str:
        """构造多模态 Prompt，整合文本和图像信息。

        Args:
            text: 原始文本问题或指令
            images: 从论文中提取的图片列表
            task: 任务类型 (analysis / description / extraction)
        """
        parts = [f"## 任务\n{text}\n"]

        if images:
            parts.append("## 从论文中提取的图表信息\n")
            for idx, img in enumerate(images[:10], 1):
                desc = f"### 图表 {idx}（第 {img.page} 页，{img.chart_type}）"
                if img.ocr_text:
                    desc += f"\n识别文字：{img.ocr_text[:500]}"
                desc += f"\n尺寸：{img.width}x{img.height}，大小：{img.size_kb:.1f}KB"
                parts.append(desc)

        task_instructions = {
            "analysis": "请结合文本和图表信息，进行深入的学术分析。",
            "description": "请描述这些图表的内容和它们在论文中的作用。",
            "extraction": "请从这些图表中提取关键数据和结论。",
        }
        parts.append(f"\n## 指导\n{task_instructions.get(task, task_instructions['analysis'])}")

        return "\n\n".join(parts)

    def describe_images_for_llm(self, images: list[ExtractedImage]) -> str:
        """将图片信息转化为文本描述，供 LLM 理解。"""
        if not images:
            return "未从论文中检测到图表。"

        descriptions = []
        for idx, img in enumerate(images, 1):
            parts = [f"[图表 {idx}] 类型={img.chart_type}，位于第 {img.page} 页"]
            if img.ocr_text:
                parts.append(f"  识别文字：{img.ocr_text[:300]}")
            if img.chart_type == "table":
                parts.append("  → 该图表包含数据表格，可能包含实验结果对比")
            elif img.chart_type == "chart":
                parts.append("  → 该图表包含数据可视化，可能展示趋势或性能曲线")
            elif img.chart_type == "diagram":
                parts.append("  → 该图表为架构/流程图，可能展示方法框架")
            descriptions.append("\n".join(parts))

        return "\n\n".join(descriptions)

    # ── 批量处理 ────────────────────────────────────────────────────────

    def process_paper_images(self, pdf_path: Path, max_images: int = 10) -> dict:
        """处理单篇论文的图表，返回结构化结果。"""
        images = self.extract_images_from_pdf(pdf_path, max_images)

        by_type: dict[str, list[ExtractedImage]] = {}
        for img in images:
            by_type.setdefault(img.chart_type, []).append(img)

        return {
            "path": str(pdf_path),
            "total_images": len(images),
            "by_type": {k: len(v) for k, v in by_type.items()},
            "descriptions": self.describe_images_for_llm(images),
            "images": images,
        }
