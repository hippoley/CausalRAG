import sys

from branchpoint import cli


def _parse(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["branchpoint", *args])
    return cli.parse_args()


def test_agent_cli_defaults_to_hosted_lightweight_embeddings(monkeypatch):
    args = _parse(monkeypatch, "agent", "--goal", "inspect ventilation")
    assert args.embedding_provider == "openai"
    assert args.embedding_model == "text-embedding-3-small"
    assert args.vector_backend == "memory"


def test_index_cli_accepts_local_embeddings_and_faiss(monkeypatch):
    args = _parse(
        monkeypatch,
        "index",
        "--input",
        "docs",
        "--output",
        "index",
        "--embedding-provider",
        "local",
        "--embedding-model",
        "all-MiniLM-L6-v2",
        "--vector-backend",
        "faiss",
    )
    assert args.embedding_provider == "local"
    assert args.embedding_model == "all-MiniLM-L6-v2"
    assert args.vector_backend == "faiss"


def test_probe_cli_has_distinct_human_in_the_loop_server(monkeypatch):
    args = _parse(monkeypatch, "probe", "--port", "9876", "--open")
    assert args.command == "probe"
    assert args.host == "127.0.0.1"
    assert args.port == 9876
    assert args.open is True
