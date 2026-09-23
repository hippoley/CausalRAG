from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path


REQUIRED_WHEEL_ENTRIES = {
    "branchpoint/__init__.py",
    "branchpoint/authorization.py",
    "branchpoint/execution.py",
    "branchpoint/postgres_execution.py",
    "branchpoint/integrations/__init__.py",
    "branchpoint/integrations/openai_agents.py",
    "branchpoint/integrations/mcp.py",
    "branchpoint/jev.py",
    "branchpoint/templates/probe_landing.html",
    "branchpoint/templates/agent_workbench.html",
    "branchpoint/templates/playable_probe.html",
}

REQUIRED_EXTRAS = {
    "api",
    "dev",
    "evaluation",
    "faiss",
    "full",
    "local-embeddings",
    "mcp",
    "observability",
    "openai-agents",
    "postgres",
    "retrieval",
}

REQUIRED_OPTIONAL_DEPENDENCIES = {
    "mcp": "mcp",
    "openai-agents": "openai-agents",
    "postgres": "psycopg",
}


def source_version(repository_root: Path) -> str:
    text = (repository_root / "branchpoint" / "__init__.py").read_text(
        encoding="utf-8"
    )
    match = re.search(
        r'^__version__ = ["\']([^"\']+)["\']',
        text,
        re.MULTILINE,
    )
    if not match:
        raise SystemExit("Could not read branchpoint.__version__")
    return match.group(1)


def metadata_value(metadata: str, key: str) -> str:
    prefix = f"{key}: "
    for line in metadata.splitlines():
        if line.startswith(prefix):
            return line[len(prefix) :].strip()
    raise SystemExit(f"Wheel metadata is missing {key!r}")


def all_metadata_values(metadata: str, key: str) -> list[str]:
    prefix = f"{key}: "
    return [
        line[len(prefix) :].strip()
        for line in metadata.splitlines()
        if line.startswith(prefix)
    ]


def verify_wheel(dist_dir: Path, repository_root: Path) -> Path:
    wheels = sorted(dist_dir.glob("branchpoint-*.whl"))
    if len(wheels) != 1:
        raise SystemExit(
            f"Expected exactly one branchpoint wheel in {dist_dir}, found {len(wheels)}"
        )
    wheel = wheels[0]

    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        missing = sorted(REQUIRED_WHEEL_ENTRIES - names)
        if missing:
            raise SystemExit(f"Wheel is missing runtime assets: {missing}")

        metadata_paths = [
            name
            for name in names
            if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_paths) != 1:
            raise SystemExit(
                "Expected exactly one .dist-info/METADATA file in wheel"
            )
        metadata = archive.read(metadata_paths[0]).decode("utf-8")

    if metadata_value(metadata, "Name").lower() != "branchpoint":
        raise SystemExit("Wheel distribution name is not 'branchpoint'")

    expected_version = source_version(repository_root)
    wheel_version = metadata_value(metadata, "Version")
    if wheel_version != expected_version:
        raise SystemExit(
            f"Wheel/source version mismatch: wheel={wheel_version} source={expected_version}"
        )

    extras = set(all_metadata_values(metadata, "Provides-Extra"))
    missing_extras = sorted(REQUIRED_EXTRAS - extras)
    if missing_extras:
        raise SystemExit(
            f"Wheel metadata is missing optional extras: {missing_extras}"
        )

    requirements = all_metadata_values(metadata, "Requires-Dist")
    for extra, dependency in REQUIRED_OPTIONAL_DEPENDENCIES.items():
        marker = re.compile(
            rf"""extra\s*==\s*["']{re.escape(extra)}["']""",
            re.IGNORECASE,
        )
        matching = [
            requirement
            for requirement in requirements
            if requirement.lower().startswith(dependency.lower())
            and marker.search(requirement)
        ]
        if not matching:
            raise SystemExit(
                f"Wheel metadata does not bind {dependency!r} to extra {extra!r}"
            )

    print(
        f"verified {wheel.name}: version={wheel_version} "
        f"entries={len(names)} extras={len(extras)}"
    )
    return wheel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a built Branchpoint wheel before release."
    )
    parser.add_argument(
        "dist_dir",
        nargs="?",
        default="dist",
        type=Path,
    )
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    verify_wheel(args.dist_dir.resolve(), repository_root)


if __name__ == "__main__":
    main()
