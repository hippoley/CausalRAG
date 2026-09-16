#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Command-line interface for CausalRAG."""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List

from causalrag import CausalRAGPipeline, __version__, create_agent, create_pipeline
from causalrag.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def _load_documents(path: str) -> List[str]:
    source = Path(path)
    files = sorted(source.rglob("*.txt")) if source.is_dir() else [source]
    documents = []
    for file_path in files:
        if file_path.exists() and file_path.is_file():
            text = file_path.read_text(encoding="utf-8").strip()
            if text:
                documents.append(text)
    return documents


def parse_args():
    parser = argparse.ArgumentParser(
        description="CausalRAG: a causal world-model runtime for goal-directed agents"
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    agent_parser = subparsers.add_parser("agent", help="Run the v0.2 causal agent")
    agent_parser.add_argument("--goal", "-g", required=True, help="Goal for the agent")
    agent_parser.add_argument("--input", "-i", help="Optional .txt file or directory to index")
    agent_parser.add_argument("--index", help="Optional persistent index directory")
    agent_parser.add_argument("--model", default="gpt-4o-mini", help="Reasoning model")
    agent_parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "local"])
    agent_parser.add_argument("--max-steps", type=int, default=8)
    agent_parser.add_argument("--json", action="store_true", help="Print complete JSON trace")

    index_parser = subparsers.add_parser("index", help="Build a reusable causal/vector index")
    index_parser.add_argument("--input", "-i", required=True)
    index_parser.add_argument("--output", "-o", required=True)
    index_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")

    query_parser = subparsers.add_parser("query", help="Use the legacy one-shot causal RAG path")
    query_parser.add_argument("--index", "-i", required=True)
    query_parser.add_argument("--query", "-q", required=True)
    query_parser.add_argument("--model", default="gpt-4o-mini")
    query_parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "local"])
    query_parser.add_argument("--top-k", type=int, default=5)

    serve_parser = subparsers.add_parser("serve", help="Start API server with /agent/run")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(level=logging.INFO)

    if args.version:
        print(f"CausalRAG version {__version__}")
        return 0

    if args.command == "agent":
        documents = _load_documents(args.input) if args.input else None
        graph_path = None
        if args.index:
            candidate = os.path.join(args.index, "causal_graph.json")
            graph_path = candidate if os.path.exists(candidate) else None
        agent = create_agent(
            model_name=args.model,
            provider=args.provider,
            graph_path=graph_path,
            index_path=args.index,
            documents=documents,
        )
        result = agent.run(args.goal, max_steps=args.max_steps)
        payload = result.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(payload["answer"])
            print(f"\nsteps={payload['steps']} stop_reason={payload['stop_reason']}")
        return 0

    if args.command == "index":
        documents = _load_documents(args.input)
        if not documents:
            logger.error("No non-empty .txt documents found")
            return 1
        pipeline = CausalRAGPipeline(
            embedding_model=args.embedding_model,
            index_path=args.output,
            provider="local",
        )
        pipeline.index(documents)
        print(f"Indexed {len(documents)} documents into {args.output}")
        return 0

    if args.command == "query":
        graph_path = os.path.join(args.index, "causal_graph.json")
        pipeline = create_pipeline(
            model_name=args.model,
            provider=args.provider,
            graph_path=graph_path if os.path.exists(graph_path) else None,
            index_path=args.index,
        )
        result = pipeline.run(args.query, top_k=args.top_k)
        print(result["answer"])
        return 0

    if args.command == "serve":
        import uvicorn

        uvicorn.run(
            "causalrag.interface.agent_api:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
        return 0

    print("No command specified. Use --help for usage information.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
