#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Command-line interface for Branchpoint."""

import argparse
import importlib
import importlib.util
import json
import logging
import os
import sys
import threading
import webbrowser
from pathlib import Path
from typing import List

from branchpoint import __version__, create_agent, create_pipeline
from branchpoint.decision_io import DecisionPayloadError, arbitrate_payload
from branchpoint.utils.logging import setup_logging

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


def _load_pack_modules(specs):
    loaded = []
    for pack_index, raw in enumerate(specs or []):
        value = str(raw or "").strip()
        if not value:
            continue

        target = value
        callable_name = ""
        if ":" in value:
            maybe_target, maybe_callable = value.rsplit(":", 1)
            if maybe_callable.isidentifier():
                target = maybe_target
                callable_name = maybe_callable

        target = target.strip()
        if not target:
            raise RuntimeError("--load-pack requires module/file or target:callable")

        source = Path(target).expanduser()
        if source.suffix == ".py":
            if not source.exists() or not source.is_file():
                raise RuntimeError(f"Capability pack file not found: {target}")
            module_name = f"_branchpoint_pack_{pack_index}_{source.stem}"
            spec = importlib.util.spec_from_file_location(module_name, source.resolve())
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Could not load capability pack file: {target}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                sys.modules.pop(module_name, None)
                raise RuntimeError(
                    f"Could not import capability pack file {target!r}: {exc}"
                ) from exc
        else:
            try:
                module = importlib.import_module(target)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not import capability pack module {target!r}: {exc}"
                ) from exc

        if callable_name:
            factory = getattr(module, callable_name, None)
            if not callable(factory):
                raise RuntimeError(
                    f"Capability pack loader {value!r} is not callable"
                )
            try:
                factory()
            except Exception as exc:
                raise RuntimeError(
                    f"Capability pack loader {value!r} failed: {exc}"
                ) from exc
        loaded.append(value)
    return loaded


def _load_decision_payload(path: str):
    if path == "-":
        raw = sys.stdin.read()
    else:
        source = Path(path)
        if not source.exists() or not source.is_file():
            raise DecisionPayloadError(f"decision file not found: {path}")
        raw = source.read_text(encoding="utf-8")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DecisionPayloadError(
            f"decision input must be valid JSON: line {exc.lineno} column {exc.colno}"
        ) from exc
    if not isinstance(payload, dict):
        raise DecisionPayloadError("decision input root must be a JSON object")
    return payload


def _require_api():
    try:
        import uvicorn
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "The HTTP server requires optional API dependencies. "
            "Install them with: pip install 'branchpoint[api]'"
        ) from exc
    return uvicorn


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--embedding-provider",
        default="openai",
        choices=["openai", "local"],
        help="Embedding provider. 'local' requires branchpoint[local-embeddings].",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Embedding model name for the selected provider.",
    )
    parser.add_argument(
        "--vector-backend",
        default="memory",
        choices=["memory", "faiss"],
        help="Vector backend. 'faiss' requires branchpoint[faiss].",
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Branchpoint: a causal decision runtime with explicit world models"
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    packs_parser = subparsers.add_parser(
        "packs",
        help="List registered capability packs available to the probe runtime",
    )
    packs_parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable capability-pack metadata.",
    )
    packs_parser.add_argument(
        "--load-pack",
        action="append",
        default=[],
        metavar="MODULE|FILE[:CALLABLE]",
        help=(
            "Load an application capability-pack module or .py file before listing. "
            "Repeat for multiple packs."
        ),
    )

    decide_parser = subparsers.add_parser(
        "decide",
        help="Arbitrate one bounded decision from a portable JSON payload",
    )
    decide_parser.add_argument(
        "input",
        help="Decision JSON file, or '-' to read JSON from stdin.",
    )
    decide_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the complete ranking, score provenance, and truthfulness metadata.",
    )

    agent_parser = subparsers.add_parser("agent", help="Run the v0.3 causal agent runtime")
    agent_parser.add_argument("--goal", "-g", required=True, help="Goal for the agent")
    agent_parser.add_argument("--input", "-i", help="Optional .txt file or directory to index")
    agent_parser.add_argument("--index", help="Optional persistent index directory")
    agent_parser.add_argument("--model", default="gpt-5.6-terra", help="Reasoning model")
    agent_parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "local"])
    agent_parser.add_argument("--max-steps", type=int, default=8)
    agent_parser.add_argument("--json", action="store_true", help="Print complete JSON trace")
    _add_embedding_args(agent_parser)

    index_parser = subparsers.add_parser("index", help="Build a reusable causal/vector index")
    index_parser.add_argument("--input", "-i", required=True)
    index_parser.add_argument("--output", "-o", required=True)
    _add_embedding_args(index_parser)

    query_parser = subparsers.add_parser("query", help="Use the legacy one-shot causal retrieval path")
    query_parser.add_argument("--index", "-i", required=True)
    query_parser.add_argument("--query", "-q", required=True)
    query_parser.add_argument("--model", default="gpt-5.6-terra")
    query_parser.add_argument("--provider", default="openai", choices=["openai", "anthropic", "local"])
    query_parser.add_argument("--top-k", type=int, default=5)
    _add_embedding_args(query_parser)

    serve_parser = subparsers.add_parser("serve", help="Start API server with /agent/run")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)

    probe_parser = subparsers.add_parser(
        "probe",
        help="Start the human-in-the-loop Playable Causal Probe",
    )
    probe_parser.add_argument("--host", default="127.0.0.1")
    probe_parser.add_argument("--port", type=int, default=8765)
    probe_parser.add_argument(
        "--open",
        action="store_true",
        help="Open the Playable Probe in the default browser after startup.",
    )
    probe_parser.add_argument(
        "--load-pack",
        action="append",
        default=[],
        metavar="MODULE[:CALLABLE]",
        help=(
            "Load/register an application capability-pack module or .py file before the server starts. "
            "Repeat for multiple packs."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging()

    if args.version:
        print(f"Branchpoint version {__version__}")
        return 0

    if args.command == "packs":
        try:
            _load_pack_modules(args.load_pack)
        except RuntimeError as exc:
            logger.error(str(exc))
            return 2
        from branchpoint.probe import available_probe_config

        rows = available_probe_config()["scenarios"]
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for row in rows:
                print(
                    f"{row['id']:<25} "
                    f"{row.get('label', row['id'])}"
                )
                description = str(row.get("description") or "").strip()
                if description:
                    print(f"  {description}")
        return 0

    if args.command == "decide":
        try:
            result = arbitrate_payload(_load_decision_payload(args.input))
        except DecisionPayloadError as exc:
            logger.error(str(exc))
            return 2
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
        else:
            proposer = result["proposer_first"]["name"]
            selected = result["selected"]["name"]
            changed = result["changed_proposer_order"]
            marker = "changed" if changed else "accepted"
            print(f"proposer: {proposer}")
            print(f"runtime:  {selected}")
            print(f"branch:   {marker}")
            if result["reasons"]:
                print("why:")
                for reason in result["reasons"]:
                    code = reason.get("code", "runtime_reason")
                    if code == "canonical_tool_policy":
                        fields = ", ".join(reason.get("fields") or [])
                        print(f"  - canonical tool policy: {fields}")
                    elif code == "higher_runtime_utility":
                        print(
                            "  - higher runtime utility: "
                            f"{float(reason.get('delta', 0.0)):+.3f}"
                        )
                    elif code == "runtime_information_source":
                        print(
                            "  - information source: "
                            f"{reason.get('source', 'runtime')}"
                        )
                    else:
                        print(f"  - {code}")
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
            embedding_provider_name=args.embedding_provider,
            embedding_model=args.embedding_model,
            vector_backend=args.vector_backend,
        )
        result = agent.run(args.goal, max_steps=args.max_steps)
        payload = result.to_dict()
        if args.json:
            print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
        else:
            print(payload["answer"])
            print(
                f"\nsteps={payload['steps']} "
                f"executed_actions={payload['executed_actions']} "
                f"stop_reason={payload['stop_reason']}"
            )
        return 0

    if args.command == "index":
        documents = _load_documents(args.input)
        if not documents:
            logger.error("No non-empty .txt documents found")
            return 1
        pipeline = create_pipeline(
            embedding_model=args.embedding_model,
            embedding_provider_name=args.embedding_provider,
            vector_backend=args.vector_backend,
            index_path=args.output,
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
            embedding_provider_name=args.embedding_provider,
            embedding_model=args.embedding_model,
            vector_backend=args.vector_backend,
        )
        result = pipeline.run(args.query, top_k=args.top_k)
        print(result["answer"])
        return 0

    if args.command == "serve":
        uvicorn = _require_api()
        uvicorn.run(
            "branchpoint.interface.agent_api:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
        return 0

    if args.command == "probe":
        try:
            _load_pack_modules(args.load_pack)
        except RuntimeError as exc:
            logger.error(str(exc))
            return 2
        uvicorn = _require_api()
        if args.open:
            browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
            url = f"http://{browser_host}:{args.port}/"
            threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        uvicorn.run(
            "branchpoint.interface.probe_api:app",
            host=args.host,
            port=args.port,
            log_level="info",
        )
        return 0

    print("No command specified. Use --help for usage information.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
