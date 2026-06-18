from __future__ import annotations

import argparse
from pathlib import Path

from cityscholar.agent import CityScholarAgent
from cityscholar.config import AppConfig
from cityscholar.memory import MemoryStore
import sys
import logging


# basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="CityScholar-Agent",
        description="Local paper RAG assistant for academic reading and review preparation.",
    )
    parser.add_argument("--storage", default="storage", help="Directory for the local index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_cmd = subparsers.add_parser("build", help="Build local paper knowledge base.")
    build_cmd.add_argument("--papers", default="papers", help="Directory containing PDF/TXT/MD papers.")
    build_cmd.add_argument("--chunk-size", type=int, default=900)
    build_cmd.add_argument("--overlap", type=int, default=120)

    ask_cmd = subparsers.add_parser("ask", help="Ask a question over the local paper knowledge base.")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--top-k", type=int, default=5)

    analyze_cmd = subparsers.add_parser("analyze", help="Analyze one paper matched by filename keyword or file path.")
    analyze_cmd.add_argument("paper_keyword", nargs="?", help="Filename keyword to match a paper.")
    analyze_cmd.add_argument("--file", dest="paper_file", help="Path to a specific paper file to analyze.")
    analyze_cmd.add_argument("--title", dest="paper_title", help="Paper title (will attempt substring or semantic matching).")
    analyze_cmd.add_argument("--save-memory", dest="save_memory", action="store_true", help="Save the analysis into the local memory store.")

    compare_cmd = subparsers.add_parser("compare", help="Compare multiple papers by filename keywords or file paths.")
    compare_cmd.add_argument("paper_keywords", nargs="*", help="Filename keywords to match papers.")
    compare_cmd.add_argument("--files", dest="paper_files", nargs="*", help="Paths to specific paper files to compare.")

    outline_cmd = subparsers.add_parser("outline", help="Generate a literature review outline.")
    outline_cmd.add_argument("topic")

    memory_cmd = subparsers.add_parser("memory", help="Query or list memory entries.")
    memory_cmd.add_argument("keyword", nargs="?", help="Keyword to search memory; omit to list all entries.")
    memory_cmd.add_argument("--limit", type=int, default=10)

    faiss_cmd = subparsers.add_parser("faiss", help="FAISS index utilities.")
    faiss_cmd.add_argument("--rebuild", action="store_true", help="Rebuild and persist FAISS index from existing embeddings.")

    papers_cmd = subparsers.add_parser("papers", help="List all indexed papers or search for paper names.")
    papers_cmd.add_argument("search_query", nargs="?", help="Optional search keyword to filter papers by title.")
    papers_cmd.add_argument("--limit", type=int, default=50, help="Max number of papers to show.")

    workflow_cmd = subparsers.add_parser("workflow", help="Run a full research assistant workflow.")
    workflow_cmd.add_argument("topic")
    workflow_cmd.add_argument("--top-k", type=int, default=6)

    multiagent_cmd = subparsers.add_parser("multiagent", help="Run multi-agent collaborative analysis.")
    multiagent_cmd.add_argument("query")
    multiagent_cmd.add_argument("--top-k", type=int, default=5)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = AppConfig.from_env(storage_dir=Path(args.storage))
    agent = CityScholarAgent(config)

    if args.command == "build":
        summary = agent.build(Path(args.papers), args.chunk_size, args.overlap)
        print(summary)
    elif args.command == "faiss":
        # rebuild persistent FAISS index from embeddings
        idx = agent.indexer
        if getattr(args, 'rebuild', False):
            # check faiss availability first to provide actionable hint
            try:
                import faiss  # type: ignore
                msg = idx.rebuild_faiss()
                print(msg)
            except Exception:
                print("FAISS 未安装或不可用。可通过 Conda 安装（推荐）：\n  conda install -c pytorch faiss-cpu\n或（在支持的系统上）使用 pip：\n  pip install faiss-cpu")
        else:
            print("Use --rebuild to (re)generate and persist the FAISS index from embeddings.")
    elif args.command == "ask":
        print(agent.answer(args.question, args.top_k).to_markdown())
    elif args.command == "papers":
        query = getattr(args, 'search_query', None)
        limit = getattr(args, 'limit', 50)
        if query:
            candidates = agent.find_paper_candidates(query, max_candidates=limit)
            if candidates:
                print(f"\n匹配 \"{query}\" 的论文（共 {len(candidates)} 篇）：\n")
                for i, title in enumerate(candidates, 1):
                    print(f"  [{i}] {title}")
                print(f"\n提示：使用以下命令分析某篇论文：")
                print(f'  python main.py analyze --title "{candidates[0][:30]}..."')
            else:
                print(f"未找到匹配 \"{query}\" 的论文。")
        else:
            all_papers = agent.list_papers()
            print(f"\n知识库中所有论文（共 {len(all_papers)} 篇）：\n")
            for i, title in enumerate(all_papers, 1):
                print(f"  [{i:2d}] {title}")
            print(f"\n提示：使用 python main.py papers \"关键词\" 过滤论文列表。")
    elif args.command == "analyze":
        # If no specific match criteria, show candidates first
        keyword = getattr(args, 'paper_keyword', None)
        file_path = Path(args.paper_file) if getattr(args, 'paper_file', None) else None
        title_query = getattr(args, 'paper_title', None)
        if not keyword and not file_path and not title_query:
            print("请指定论文匹配条件。使用方式：")
            print('  python main.py analyze "关键词"           # 按标题关键词匹配')
            print('  python main.py analyze --title "完整标题"  # 按完整标题匹配')
            print('  python main.py analyze --file path.pdf     # 按文件路径匹配')
            print('\n提示：先运行 python main.py papers "关键词" 查看匹配的论文列表。')
        else:
            # Show candidates for user confirmation
            search_key = keyword or title_query or ""
            if search_key and not file_path:
                candidates = agent.find_paper_candidates(search_key, max_candidates=8)
                if len(candidates) > 1:
                    print(f"\n找到 {len(candidates)} 篇匹配论文：")
                    for i, title in enumerate(candidates, 1):
                        print(f"  [{i}] {title}")
                    print(f"\n将分析第一篇：{candidates[0]}")
                    print("如需分析其他论文，请使用 --title 参数指定完整标题。\n")
            result = agent.analyze_paper(
                keyword=keyword,
                file_path=file_path,
                title_query=title_query,
                save_memory=getattr(args, "save_memory", False),
            )
            print(result.to_markdown())
    elif args.command == "compare":
        files = [Path(p) for p in args.paper_files] if getattr(args, "paper_files", None) else None
        print(agent.compare_papers(keywords=args.paper_keywords or [], file_paths=files).to_markdown())
    elif args.command == "outline":
        print(agent.generate_outline(args.topic).to_markdown())
    elif args.command == "workflow":
        report_path = agent.run_workflow(args.topic, args.top_k)
        print(f"Markdown report exported: {report_path}")
    elif args.command == "multiagent":
        from cityscholar.multiagent import MultiAgentHub
        retriever = agent._retriever()
        hub = MultiAgentHub()
        result = hub.run_analysis_workflow(
            query=args.query,
            retriever=retriever,
            llm_client=agent.llm,
            top_k=args.top_k,
        )
        print(result.to_markdown())
    elif args.command == "memory":
        # use agent.memory if available, otherwise create MemoryStore
        mem = getattr(agent, "memory", None) or MemoryStore(config.storage_dir)
        if args.keyword:
            hits = mem.query(args.keyword, limit=args.limit)
        else:
            hits = mem.all()[: args.limit]

        if not hits:
            print("No memory entries found.")
        else:
            for idx, e in enumerate(hits, 1):
                print(f"[{idx}] type={e.type} ts={e.ts} metadata={e.metadata}")
                try:
                    print(e.content)
                except UnicodeEncodeError:
                    # fallback: write bytes directly to stdout buffer to avoid encoding errors
                    try:
                        sys.stdout.buffer.write((e.content + "\n").encode("utf-8", errors="replace"))
                    except Exception:
                        # final fallback: print a summary
                        print("[Unprintable content]")
                print("---")


if __name__ == "__main__":
    main()
