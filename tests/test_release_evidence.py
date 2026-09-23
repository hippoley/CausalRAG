from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_release_evidence import build_release_evidence


ROOT = Path(__file__).resolve().parents[1]


def _doctor(path: Path, *, ok: bool = True, version: str = "0.3.0") -> Path:
    payload = {
        "schema_version": "branchpoint.doctor.v1",
        "branchpoint_version": version,
        "ok": ok,
        "checks": [
            {"name": "decision_runtime", "status": "ok"},
            {"name": "authorization", "status": "ok"},
            {"name": "durable_receipt", "status": "ok"},
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_release_evidence_binds_artifact_hashes_and_doctor(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    wheel = dist / "branchpoint-0.3.0-py3-none-any.whl"
    sdist = dist / "branchpoint-0.3.0.tar.gz"
    wheel.write_bytes(b"wheel-bytes")
    sdist.write_bytes(b"sdist-bytes")

    payload = build_release_evidence(
        repository_root=ROOT,
        dist_dir=dist,
        doctor_path=_doctor(tmp_path / "doctor.json"),
        source_commit="abc123",
        source_ref="v0.3.0",
    )

    assert payload["schema_version"] == "branchpoint.release-evidence.v1"
    assert payload["package"] == {"name": "branchpoint", "version": "0.3.0"}
    assert payload["source"] == {
        "repository": "hippoley/CausalRAG",
        "commit": "abc123",
        "ref": "v0.3.0",
    }
    assert payload["doctor"]["ok"] is True

    by_name = {row["name"]: row for row in payload["artifacts"]}
    assert by_name[wheel.name]["sha256"] == hashlib.sha256(b"wheel-bytes").hexdigest()
    assert by_name[sdist.name]["sha256"] == hashlib.sha256(b"sdist-bytes").hexdigest()
    assert by_name[wheel.name]["bytes"] == len(b"wheel-bytes")


def test_release_evidence_rejects_failed_doctor(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "branchpoint-0.3.0-py3-none-any.whl").write_bytes(b"wheel")
    (dist / "branchpoint-0.3.0.tar.gz").write_bytes(b"sdist")

    with pytest.raises(SystemExit, match="Doctor evidence is not successful"):
        build_release_evidence(
            repository_root=ROOT,
            dist_dir=dist,
            doctor_path=_doctor(tmp_path / "doctor.json", ok=False),
        )


def test_release_evidence_rejects_version_mismatch(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "branchpoint-0.3.0-py3-none-any.whl").write_bytes(b"wheel")
    (dist / "branchpoint-0.3.0.tar.gz").write_bytes(b"sdist")

    with pytest.raises(SystemExit, match="Doctor/package version mismatch"):
        build_release_evidence(
            repository_root=ROOT,
            dist_dir=dist,
            doctor_path=_doctor(
                tmp_path / "doctor.json",
                version="9.9.9",
            ),
        )


def test_release_evidence_rejects_distribution_filename_version_mismatch(tmp_path):
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "branchpoint-9.9.9-py3-none-any.whl").write_bytes(b"wheel")
    (dist / "branchpoint-0.3.0.tar.gz").write_bytes(b"sdist")

    with pytest.raises(SystemExit, match="filename/version mismatch"):
        build_release_evidence(
            repository_root=ROOT,
            dist_dir=dist,
            doctor_path=_doctor(tmp_path / "doctor.json"),
        )
