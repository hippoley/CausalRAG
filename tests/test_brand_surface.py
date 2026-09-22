import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PUBLIC_FILES = [
    ROOT / "README.MD",
    ROOT / "SECURITY.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CITATION.cff",
    ROOT / "setup.py",
    ROOT / "examples" / "README.md",
    ROOT / "tests" / "README.md",
]

PUBLIC_DIRS = [
    ROOT / "docs",
    ROOT / "site",
    ROOT / ".github" / "ISSUE_TEMPLATE",
    ROOT / "branchpoint" / "templates",
]

TEXT_SUFFIXES = {".md", ".html", ".yml", ".yaml", ".py", ".cff", ".svg"}

# The Python import/CLI remains `branchpoint` for compatibility. What is retired is
# the old public *brand* and external repository namespace.
RETIRED_REPOSITORY_NAMESPACE = re.compile(r"branchpoint-runtime", re.IGNORECASE)
RETIRED_PUBLIC_BRAND = re.compile(r"\bBranchpoint\b")
RETIRED_EVALUATOR = re.compile(r"\bragas\b", re.IGNORECASE)


def _public_text_files():
    seen = set()
    for path in PUBLIC_FILES:
        if path.is_file() and path not in seen:
            seen.add(path)
            yield path
    for directory in PUBLIC_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES and path not in seen:
                seen.add(path)
                yield path


def test_public_brand_is_causalrag():
    violations = []
    for path in _public_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        reasons = []
        if RETIRED_REPOSITORY_NAMESPACE.search(text):
            reasons.append("retired repository namespace")
        if RETIRED_PUBLIC_BRAND.search(text):
            reasons.append("retired public brand")
        if RETIRED_EVALUATOR.search(text):
            reasons.append("retired evaluator")
        if reasons:
            violations.append(f"{path.relative_to(ROOT)}: {', '.join(reasons)}")
    assert not violations, "\n".join(violations)


def test_primary_surfaces_name_causalrag():
    readme = (ROOT / "README.MD").read_text(encoding="utf-8")
    landing = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    examples = (ROOT / "site" / "examples.html").read_text(encoding="utf-8")

    assert "# CausalRAG" in readme
    assert "<title>CausalRAG" in landing
    assert "<title>CausalRAG" in examples
    assert "https://hippoley.github.io/CausalRAG/" in landing
