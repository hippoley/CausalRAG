import time

from branchpoint.probe import ProbeRunConfig, ProbeSessionManager, SQLiteSessionArchive


def _wait_for(session, status, timeout=6.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if session.status() == status:
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {status}: {session.snapshot()}")


def _finish(session, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = session.status()
        if status == "waiting_for_human":
            session.resolve_decision("approve")
        elif status == "completed":
            return
        elif status == "failed":
            raise AssertionError(session.snapshot().get("error"))
        time.sleep(0.01)
    raise AssertionError(f"session did not complete: {session.snapshot()}")


def test_archive_persists_pending_gate_and_survives_reopen(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    archive = SQLiteSessionArchive(path)
    manager = ProbeSessionManager(archive)
    session = manager.create(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    _wait_for(session, "waiting_for_human")

    deadline = time.time() + 3.0
    payload = None
    while time.time() < deadline:
        try:
            payload = SQLiteSessionArchive(path).get(session.session_id)
        except KeyError:
            payload = None
        if payload and payload.get("pending_decision"):
            break
        time.sleep(0.01)

    assert payload is not None
    assert payload["status"] == "waiting_for_human"
    assert payload["pending_decision"]["gate_id"]
    assert payload["world_model"]["hypotheses"]

    session.gate.close()


def test_archive_records_completed_session_and_human_history(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    archive = SQLiteSessionArchive(path)
    manager = ProbeSessionManager(archive)
    session = manager.create(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            proposer_family="deterministic",
        )
    )
    _finish(session)

    payload = SQLiteSessionArchive(path).get(session.session_id)
    assert payload["status"] == "completed"
    assert payload["result"]["metrics"]["success"] is True
    assert payload["episode_ledger"]
    assert payload["interactions"]["human_gate_history"]
    assert payload["trace"]


def test_manager_can_read_archive_without_live_session(tmp_path):
    path = tmp_path / "sessions.sqlite3"
    archive = SQLiteSessionArchive(path)
    archive.save(
        {
            "session_id": "s-1",
            "status": "completed",
            "config": {"proposer_family": "deterministic"},
            "episode_ledger": [{"step": 0}],
        }
    )

    restarted_manager = ProbeSessionManager(SQLiteSessionArchive(path))
    payload = restarted_manager.archived("s-1")
    assert payload["session_id"] == "s-1"
    assert payload["episode_ledger"] == [{"step": 0}]
