# CityScholar-Agent

CityScholar-Agent 是一个面向学术论文阅读与综述准备的本地科研助教 Agent。项目可以读取本地 PDF 论文，构建本地论文知识库，并支持检索问答、单篇论文分析、多篇论文比较、综述提纲生成和 Markdown 报告导出。

---

## 1. 项目功能

- 本地论文知识库：读取 `papers/` 目录中的 PDF / TXT / MD 文件，切分文本片段并保存索引。
- RAG 检索问答：围绕用户问题检索相关论文片段，并基于证据生成回答。
- 混合检索：结合 BM25 关键词检索与向量检索，通过 Reciprocal Rank Fusion 融合排序。
- 单篇论文分析：输出研究问题、方法、主要发现、局限性和可引用证据。
- 多篇论文比较：从研究主题、方法、数据、结论和局限性等角度比较论文。
- 综述提纲生成：根据本地论文生成可扩展的文献综述结构。
- Markdown 导出：将完整科研辅助流程的结果导出到 `outputs/` 目录。
- 可选大模型调用：支持 OpenAI 兼容 API（DeepSeek、Qwen 等），未配置时使用本地抽取式生成。

---

## 2. 文件构成与模块说明

```text
CityScholar-Agent/
├── main.py                     # 命令行入口（build/ask/analyze/compare/outline/workflow/multiagent/papers）
├── requirements.txt            # 环境依赖
├── README.md                   # 项目说明文档
├── .env.local                  # 本地 API 配置（LLM_API_KEY, LLM_BASE_URL, LLM_MODEL）
├── papers/                     # 放置本地论文 PDF/TXT/MD
├── storage/                    # 自动保存向量索引和文本片段
├── outputs/                    # 自动导出 Markdown 报告
└── cityscholar/
    ├── __init__.py
    ├── config.py               # 配置读取与管理
    ├── models.py               # 数据结构定义
    ├── indexer.py              # 文档加载、切分、索引构建
    ├── retrieval.py            # 混合检索引擎（BM25 + 向量 + RRF 融合 + 重排序）
    ├── llm.py                  # 大模型 API 调用与本地生成回退
    ├── agent.py                # Agent 工作流编排（查询分解、反思、置信度评估）
    ├── utils.py                # 通用工具函数
    ├── graph_rag.py            # GraphRAG 图增强检索
    ├── multimodal.py           # 多模态能力（PDF 图表提取、OCR）
    ├── multiagent.py           # 多智能体协同（Reader/Analyst/Critic/Writer）
    ├── mcp_protocol.py         # MCP 通讯协议（工具注册、消息传递）
    ├── safety.py               # 智能体安全（注入检测、信息过滤、学术诚信）
    └── tree_search.py          # 树搜索增强（Best-first 推理搜索）
```

### 2.1 核心模块详解

#### config.py — 配置管理
- 从环境变量和 `.env.local` 文件读取配置
- 支持 LLM API 配置（Key / Base URL / Model）
- 支持检索参数（FAISS、GraphRAG、融合权重）
- 支持智能增强参数（反思轮数、查询扩展、重排序开关）

#### models.py — 数据结构
- `Chunk`：文本片段，含标题、页码、来源信息
- `SearchResult`：检索结果，含匹配分数、匹配类型和图扩展路径
- `AgentResponse`：Agent 完整响应，含回答文本、置信度、证据、推理追踪

#### indexer.py — 文档索引
- PDF 读取（pypdf + 字节流回退 + 跳过扫描件）
- 文本切分（语义段落感知、中英文混合策略）
- TF-IDF 向量索引构建（sklearn TfidfVectorizer）
- 向量索引持久化（numpy .npy 格式）
- 知识图谱构建（基于 Jaccard 相似度的语义图）

#### retrieval.py — 混合检索引擎
- **BM25 关键词评分**：词频饱和度 + 文档长度归一化
- **TF-IDF 向量检索**：余弦相似度语义匹配
- **RRF 融合排序**：Reciprocal Rank Fusion 统一排序
- **查询扩展**：30+ 组学术同义词/关联词映射（中英双语）
- **检索后重排序**：标题匹配加分、长度归一化、关键词密度、多样性惩罚
- **GraphRAG 图扩展**：基于语义图的邻域扩展检索

#### llm.py — 大模型调用
- **任务专属 System Prompt**：问答 / 论文分析 / 多论文比较 / 综述提纲 / 反思评估，每个任务独立提示词模板
- **JSON 结构化输出**：自动解析 LLM 返回的 JSON，支持多种容错策略（代码块剥离、括号匹配、常见格式修复）
- **反思生成**：评估先前回答的质量并生成改进版本
- **置信度评估**：基于 LLM 的语义评估 + 启发式回退
- **本地回退**：API 不可用时自动降级为抽取式摘要

#### agent.py — Agent 工作流编排
- **查询分解**：复杂问题自动拆解为 2-4 个子查询
- **多角度证据采集**：从不同角度检索并合并结果
- **自我反思循环**：回答生成 → 质量评审 → 修订，最多 N 轮（可配置）
- **置信度标注**：每次回答附带 0%~100% 置信度评分
- **论文检索/匹配**：按关键词、作者、年份等多种方式定位论文

### 2.2 可选扩展模块

#### graph_rag.py — GraphRAG 图增强检索
- **语义图构建**：基于 Jaccard 相似度建立论文/段落间的关联关系
- **图扩展检索**：从命中节点出发，沿图边扩展邻居，覆盖隐含关联信息
- **配置**：`graph_sim_threshold`（相似度阈值）、`graph_max_neighbors`（最大邻居数）

#### multimodal.py — 多模态能力
- **PDF 图表提取**：从 PDF 中提取嵌入的图片（基于字节流解析）
- **OCR 文字识别**：对提取的图片进行 OCR，提取表格和图表中的文字内容
- **图表类型分类**：自动识别图片是表格（table）、图表（chart）还是示意图（diagram）
- **多模态 Prompt 构造**：将 OCR 结果与文本上下文结合，生成增强的分析提示

#### multiagent.py — 多智能体协同
- **4 种专业角色**：
  - **Reader（阅读者）**：负责文献检索和信息提取
  - **Analyst（分析师）**：负责深度分析和洞察提炼
  - **Critic（评审者）**：负责质量评估和逻辑审查
  - **Writer（撰写者）**：负责结构化输出和报告撰写
- **优先级执行**：按角色优先级串行执行，前一角色的输出作为下一角色的输入
- **流水线模式**：`run_analysis_workflow` 自动执行四阶段分析流程

#### mcp_protocol.py — MCP 通讯协议
- **工具注册中心**（ToolRegistry）：统一注册和管理所有可调用工具
- **JSON Schema 描述**：每个工具都有标准的输入参数和返回值描述
- **内置工具**：search（检索）、analyze（论文分析）、compare（论文比较）、outline（提纲生成）、list_papers（论文列表）、ask（问答）
- **MCP 消息传递**（MCPNode）：支持 send / handle / list_tools 操作

#### safety.py — 智能体安全与攻防
- **Prompt 注入检测**：30+ 种注入模式识别（角色覆写、指令覆盖、编码绕过等）
- **敏感信息过滤**：自动检测并脱敏手机号、身份证号、邮箱等 PII 信息
- **学术诚信检查**：检测抄袭、伪造引用、过度借鉴等学术不端行为
- **幻觉检测**：验证回答中的引用是否真实存在于论文原文中
- **沙盒执行**：对不受信任的代码/操作进行隔离执行
- **安全事件日志**：记录所有安全相关事件，便于审计追踪

#### tree_search.py — 树搜索增强
- **推理节点**（ReasoningNode）：每个节点代表一个推理步骤，含状态、分数、深度
- **Best-first 搜索**：优先扩展最有希望的推理路径
- **Beam 剪枝**：限制同时探索的路径数，防止搜索空间爆炸
- **AgentTreeSearch 集成**：自动将复杂问题拆分为可搜索的推理步骤
- **推理路径可视化**：输出完整推理树结构

---

## 3. 环境安装

建议使用 Python 3.10 或以上版本。

```bash
cd CityScholar-Agent
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

如果暂时无法安装 `pypdf`，系统仍可处理 TXT / MD 文档；安装后即可读取 PDF 论文。

---

## 4. 配置大模型 API（可选）

项目不强制依赖在线大模型。若要使用 DeepSeek、Qwen 或其他 OpenAI 兼容接口，可创建 `.env.local` 文件：

```bash
LLM_API_KEY=你的API密钥
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

也可通过系统环境变量设置（优先级低于 `.env.local`）：

```bash
set LLM_API_KEY=你的API密钥
set LLM_BASE_URL=https://api.deepseek.com
set LLM_MODEL=deepseek-chat
```

如果不设置这些配置，Agent 会使用本地抽取式摘要和回答，仍可完成基础作业演示。

---

## 5. 运行方式

先把论文 PDF 文件复制到 `papers/` 目录，然后构建知识库：

```bash
python main.py build --papers papers --storage storage
```

### 查看/搜索知识库中的论文

```bash
python main.py papers                    # 列出所有论文
python main.py papers "差分隐私"          # 按关键词搜索
python main.py papers "王腾"              # 按作者搜索
python main.py papers "2026"              # 按年份搜索
```

### 基于本地知识库提问

```bash
python main.py ask "联邦学习中差分隐私面临哪些主要挑战？"
```

### 分析单篇论文

```bash
python main.py analyze "差分隐私"               # 关键词匹配
python main.py analyze --title "王腾" --save-memory  # 指定标题关键词并保存到记忆
python main.py analyze --file papers/xxx.pdf    # 指定文件路径
```

### 比较多篇论文

```bash
python main.py compare "隐私保护" "异构数据"
```

### 生成综述提纲

```bash
python main.py outline "城市治理场景下大语言模型智能体的研究现状"
```

### 多智能体协作分析

```bash
python main.py multiagent "联邦学习中差分隐私的主要挑战"
```

### 运行完整工作流并导出 Markdown 报告

```bash
python main.py workflow "联邦学习中的隐私保护方法"
```

导出的报告位于 `outputs/` 目录。

---

## 6. 示例结果

### 6.1 问答示例

**问题**：联邦学习中差分隐私面临哪些主要挑战？

**回答摘要**（由 DeepSeek deepseek-chat 生成，置信度 85%）：

> 基于五篇论文证据，联邦学习中差分隐私面临的主要挑战可概括为：**隐私保护与模型效能的固有权衡、数据异质性导致的性能退化、以及隐私与安全鲁棒性之间的冲突**。
>
> 1. **隐私保护与模型效能的固有权衡**：差分隐私通过向梯度注入噪声提供隐私保证，但这会直接影响模型精度。隐私预算越小（ε 越小），噪声越大，模型性能越差。
> 2. **数据异质性（Non-IID）加剧的挑战**：客户端数据的非独立同分布特性会导致客户端漂移，差分隐私的噪声会进一步干扰不稳定的本地更新，使个性化联邦学习效果大打折扣。
> 3. **隐私保护与安全鲁棒性之间的冲突**：差分隐私的噪声会掩盖恶意客户端的中毒攻击特征，降低系统对拜占庭攻击的鲁棒性。

### 6.2 完整工作流报告

运行 `python main.py workflow "联邦学习中的隐私保护方法"` 后，`outputs/` 目录会生成包含以下内容的 Markdown 报告：

- 用户研究问题
- 检索问答结果（含置信度评分和推理追踪）
- 单篇论文深度分析
- 多篇论文结构化比较
- 综述提纲（含候选题目和写作指导）
- 完整引用证据片段

---

## 7. 智能化增强特性

### 7.1 查询分解（Query Decomposition）

面对复杂学术问题，Agent 会自动将其分解为 2-4 个独立的子查询，分别检索后融合结果，确保回答全面覆盖问题各个维度。

### 7.2 查询扩展（Query Expansion）

内置 30+ 组学术同义词/关联词库（中英双语），自动为原始查询添加相关术语（如"方法" → "approach/methodology/technique"），提升检索召回率。

### 7.3 BM25 + RRF 融合检索

- 关键词检索从简单 TF 重叠升级为 BM25 风格评分（词频饱和度 + 文档长度归一化）
- 多查询结果通过 Reciprocal Rank Fusion 算法融合排序
- 检索后多维度重排序：标题匹配加分、长度归一化、关键词密度、论文多样性惩罚
- 动态权重：短查询 keyword:vector = 45:55，长查询偏向语义 35:65

### 7.4 自我反思循环（Self-Reflection）

回答生成后，Agent 会自动进行质量评审：
1. 评估证据充分性、逻辑严密性、学术规范性、完整性
2. 若评分 < 85 分，生成改进建议并自动修订
3. 最多可进行多轮反思迭代（通过 `MAX_REFLECTION_ROUNDS` 配置）

### 7.5 置信度评估

每次回答附带 0%~100% 的置信度评分，基于证据覆盖度、一致性和匹配度综合计算。

### 7.6 任务专属 System Prompt

不同任务使用不同的系统提示词模板：
- **检索问答**：强调交叉验证、区分直接证据与推断、标注知识盲区
- **论文分析**：深度批判性评价、可引用性评估、综述写作指导
- **多论文比较**：结构化比较框架、全局洞察提炼
- **综述提纲**：研究空白识别、写作指导生成

---

## 8. 项目特点

本项目完成了课程要求中的全部基础能力：
- 本地 PDF 读取与知识库构建
- RAG 增强问答与混合检索（BM25 + 向量 + RRF 融合）
- 单篇论文分析、多篇论文比较、综述提纲生成
- Markdown 报告导出

同时实现了 6 项可选扩展能力：
- GraphRAG 图增强检索
- 多模态能力（PDF 图表提取 + OCR）
- 多智能体协同（4 角色流水线）
- MCP 通讯协议（工具注册 + 消息传递）
- 智能体安全与攻防（注入检测 + 信息过滤 + 学术诚信 + 幻觉检测）
- 树搜索增强（Best-first 推理搜索）
