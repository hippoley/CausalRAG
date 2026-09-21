import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {
    ".py", ".md", ".html", ".js", ".css", ".yml", ".yaml", ".toml",
    ".txt", ".cfg", ".in", ".example", ".json",
}
EXACT_TEXT_FILES = {"Dockerfile", "LICENSE"}

LEGACY_NAMESPACE = re.compile(r"causalrag", re.IGNORECASE)
LEGACY_STANDALONE_TERM = re.compile(r"\brag\b", re.IGNORECASE)
RETIRED_EVALUATOR = re.compile(r"ragas", re.IGNORECASE)


def _text_files():
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in {".git", ".venv", "venv", "dist", "build", "__pycache__"} for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in EXACT_TEXT_FILES:
            continue
        yield path


def test_public_repository_has_no_legacy_brand_tokens():
    violations = []
    for path in _text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        reasons = []
        if LEGACY_NAMESPACE.search(text):
            reasons.append("legacy namespace")
        if LEGACY_STANDALONE_TERM.search(text):
            reasons.append("legacy standalone term")
        if RETIRED_EVALUATOR.search(text):
            reasons.append("retired evaluator")
        if reasons:
            violations.append(f"{path.relative_to(ROOT)}: {', '.join(reasons)}")
    assert not violations, "\n".join(violations)
