from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "branchpoint.release-evidence.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_version(repository_root: Path) -> str:
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


def _load_doctor(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "branchpoint.doctor.v1":
        raise SystemExit("Doctor evidence has an unsupported schema version")
    if payload.get("ok") is not True:
        raise SystemExit("Doctor evidence is not successful")
    checks = payload.get("checks")
    if not isinstance(checks, list):
        raise SystemExit("Doctor evidence checks must be a list")
    by_name = {
        row.get("name"): row
        for row in checks
        if isinstance(row, dict)
    }
    required = ("decision_runtime", "authorization", "durable_receipt")
    missing = [
        name
        for name in required
        if by_name.get(name, {}).get("status") != "ok"
    ]
    if missing:
        raise SystemExit(
            "Doctor evidence is missing successful required check(s): "
            + ", ".join(missing)
        )
    return payload


def build_release_evidence(
    *,
    repository_root: Path,
    dist_dir: Path,
    doctor_path: Path,
    source_commit: str = "",
    source_ref: str = "",
) -> dict[str, Any]:
    doctor = _load_doctor(doctor_path)
    version = _source_version(repository_root)
    if doctor.get("branchpoint_version") != version:
        raise SystemExit(
            "Doctor/package version mismatch: "
            f"doctor={doctor.get('branchpoint_version')} source={version}"
        )

    artifacts = []
    for path in sorted(dist_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".whl" or path.name.endswith(".tar.gz"):
            artifacts.append(
                {
                    "name": path.name,
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    if len(artifacts) != 2:
        raise SystemExit(
            f"Expected wheel + sdist in {dist_dir}, found {len(artifacts)} artifacts"
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "package": {
            "name": "branchpoint",
            "version": version,
        },
        "source": {
            "repository": "hippoley/CausalRAG",
            "commit": str(source_commit or ""),
            "ref": str(source_ref or ""),
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "artifacts": artifacts,
        "doctor": doctor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bind Branchpoint release artifacts to doctor evidence."
    )
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    parser.add_argument("--doctor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    payload = build_release_evidence(
        repository_root=repository_root,
        dist_dir=args.dist_dir.resolve(),
        doctor_path=args.doctor.resolve(),
        source_commit=os.environ.get("GITHUB_SHA", ""),
        source_ref=os.environ.get("GITHUB_REF_NAME", ""),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"release evidence: {args.output} "
        f"version={payload['package']['version']} "
        f"artifacts={len(payload['artifacts'])}"
    )


if __name__ == "__main__":
    main()
