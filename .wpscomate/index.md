<!-- wps-comate:index v3 -->
# Workspace Index

Updated: 2026-06-18 19:14
Root: .
Purpose: CityScholar-Agent 是一个面向学术论文阅读与综述准备的本地科研助教 Agent。项目可以读取本地 PDF 论文，构建本地论文知识库，并支持检索问.


## Stack

Python.

## Project Structure

```
C:\Users\97082\Documents\Codex\2026-05-19\files-mentioned-by-the-user-docx\CityScholar-Agent/
├── cityscholar/ — 城市学者核心包
├── outputs/ — 输出结果存储目录
├── papers/ — 论文资料存储目录
├── storage/ — 数据持久化存储目录
├── .env.example — 环境变量模板
├── README.md — 项目说明文档
├── cityscholar\agent.py — 定义AI代理的核心逻辑
├── cityscholar\config.py — 项目配置管理
├── cityscholar\graph_rag.py — 图增强检索生成
├── cityscholar\indexer.py — 数据索引构建
├── cityscholar\llm.py — 封装大型语言模型功能
├── cityscholar\mcp_protocol.py — 实现多智能体通信协议
├── cityscholar\memory.py — 内存状态管理
├── cityscholar\models.py — 数据模型定义
├── cityscholar\multiagent.py — 管理多智能体交互
├── cityscholar\multimodal.py — 处理多模态数据输入
├── cityscholar\retrieval.py — 实现信息检索功能
├── cityscholar\safety.py — 确保系统安全性
├── cityscholar\tree_search.py — 实现树搜索算法
├── cityscholar\utils.py — 通用工具函数集合
├── main.py — 应用程序主入口
└── multiagent_demo.py — 多智能体演示脚本
```

## Omitted

Detailed component files, assets, archives, changelog, generated output and lock files are omitted. Use `rg` and file reads for exact files before editing.
> This file is auto-generated. Do not edit manually.