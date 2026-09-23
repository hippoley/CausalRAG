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

# The repository URL remains /CausalRAG for compatibility until a deliberate rename.
# What must not return is the abandoned external namespace or retired evaluator.
RETIRED_REPOSITORY_NAMESPACE = re.compile(r"branchpoint-runtime", re.IGNORECASE)
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


def test_public_surfaces_have_no_retired_namespace():
    violations = []
    for path in _public_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        reasons = []
        if RETIRED_REPOSITORY_NAMESPACE.search(text):
            reasons.append("retired repository namespace")
        if RETIRED_EVALUATOR.search(text):
            reasons.append("retired evaluator")
        if reasons:
            violations.append(f"{path.relative_to(ROOT)}: {', '.join(reasons)}")
    assert not violations, "\n".join(violations)


def test_primary_surfaces_name_branchpoint():
    readme = (ROOT / "README.MD").read_text(encoding="utf-8")
    landing = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
    examples = (ROOT / "site" / "examples.html").read_text(encoding="utf-8")
    workbench = (ROOT / "branchpoint" / "templates" / "agent_workbench.html").read_text(encoding="utf-8")

    assert "# Branchpoint" in readme
    assert "<title>Branchpoint" in landing
    assert "<title>Branchpoint" in examples
    assert "<title>Branchpoint · Live Workbench" in workbench
    assert "https://hippoley.github.io/CausalRAG/" in landing


LEGACY_PRODUCT_NAME = re.compile(r"\bCausalRAG\b", re.IGNORECASE)

BRAND_IDENTITY_FILES = [
    ROOT / "README.MD",
    ROOT / "SECURITY.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "CITATION.cff",
    ROOT / "examples" / "README.md",
    ROOT / "tests" / "README.md",
    ROOT / ".github" / "ISSUE_TEMPLATE" / "showcase.yml",
    ROOT / ".github" / "ISSUE_TEMPLATE" / "failure-case.yml",
    ROOT / "docs" / "capability-packs.md",
    ROOT / "docs" / "decision-recipes.md",
    ROOT / "docs" / "runtime-discrimination.md",
    ROOT / "docs" / "v0.3-experiment-contracts.md",
    ROOT / "docs" / "v0.3-hypothesis-falsification.md",
    ROOT / "docs" / "assets" / "causalrag-hero.svg",
    ROOT / "branchpoint" / "scaffold.py",
]


def _strip_compatibility_paths(text: str) -> str:
    text = text.replace(
        "https://github.com/hippoley/CausalRAG",
        "<github-repository-url>",
    )
    text = text.replace(
        "https://hippoley.github.io/CausalRAG",
        "<github-pages-url>",
    )
    text = re.sub(
        r"\bcd\s+CausalRAG\b",
        "cd <repository-directory>",
        text,
        flags=re.IGNORECASE,
    )
    text = text.replace("/CausalRAG", "/<repository-path>")
    return text


def test_legacy_product_name_is_only_a_compatibility_path():
    violations = []
    for path in BRAND_IDENTITY_FILES:
        text = path.read_text(encoding="utf-8", errors="ignore")
        visible_identity = _strip_compatibility_paths(text)
        if LEGACY_PRODUCT_NAME.search(visible_identity):
            violations.append(str(path.relative_to(ROOT)))
    assert not violations, (
        "Legacy product identity leaked back into public surfaces:\n"
        + "\n".join(violations)
    )
