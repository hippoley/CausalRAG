import time

from fastapi.testclient import TestClient

from causalrag.interface.probe_api import app
from causalrag.probe import ARENA_MANAGER, ProbeRunConfig


client = TestClient(app)


def _wait_for(arena, predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        snap = arena.snapshot()
        if predicate(snap):
            return snap
        time.sleep(0.01)
    raise AssertionError("tester arena did not reach expected state")


def test_tester_arena_same_world_exposes_runtime_divergence_and_outcome_delta():
    arena = ARENA_MANAGER.create(
        ProbeRunConfig(
            hidden_hypothesis="H2",
            outcome_mode="deterministic",
            seed=0,
            proposer_family="deterministic",
            max_steps=6,
            max_probes=3,
        )
    )
    try:
        first = _wait_for(
            arena,
            lambda snap: (
                snap["baseline"]["status"] == "waiting_for_human"
                and snap["tester"]["status"] == "waiting_for_human"
            ),
        )
        assert first["comparison_contract"]["only_runtime_capabilities_differ"] is True
        assert first["baseline"]["config"]["seed"] == first["tester"]["config"]["seed"] == 0
        assert first["baseline"]["config"]["proposer_family"] == "deterministic"
        assert first["tester"]["config"]["proposer_family"] == "deterministic"
        assert first["baseline"]["config"]["capabilities"]["causal_selection"] is False
        assert first["tester"]["config"]["capabilities"]["causal_selection"] is True

        baseline_action = first["baseline"]["pending_decision"]["runtime_selected"]["name"]
        tester_action = first["tester"]["pending_decision"]["runtime_selected"]["name"]
        assert baseline_action != tester_action
        assert first["divergence"]["actions_differ"] is True

        for _ in range(8):
            arena.approve_waiting()
            snap = _wait_for(
                arena,
                lambda row: (
                    row["completed"]
                    or row["baseline"]["status"] == "waiting_for_human"
                    or row["tester"]["status"] == "waiting_for_human"
                ),
            )
            if snap["completed"]:
                break
            # Let both arms arrive at their next gate before approving again.
            _wait_for(
                arena,
                lambda row: (
                    row["completed"]
                    or (
                        row["baseline"]["status"] in {"waiting_for_human", "completed"}
                        and row["tester"]["status"] in {"waiting_for_human", "completed"}
                    )
                ),
            )
        final = _wait_for(arena, lambda row: row["completed"])
        assert final["baseline"]["result"]["metrics"]["success"] is False
        assert final["tester"]["result"]["metrics"]["success"] is True
        assert final["metric_delta_tester_minus_baseline"]["causal_regret"] < 0
    finally:
        ARENA_MANAGER.delete(arena.arena_id)


def test_tester_arena_http_surface_is_playable():
    page = client.get("/arena")
    assert page.status_code == 200
    assert "TESTER ARENA" in page.text
    assert "PLAIN TOOL LOOP" in page.text
    assert "CAUSAL TESTER" in page.text
    assert "Advance both one step" in page.text

    created = client.post(
        "/api/arenas",
        json={
            "hidden_hypothesis": "H2",
            "outcome_mode": "deterministic",
            "seed": 0,
            "proposer_family": "deterministic",
        },
    )
    assert created.status_code == 200
    arena_id = created.json()["arena_id"]
    try:
        snapshot = client.get(f"/api/arenas/{arena_id}")
        assert snapshot.status_code == 200
        assert snapshot.json()["comparison_contract"]["same_model"] is True

        advanced = client.post(f"/api/arenas/{arena_id}/advance")
        assert advanced.status_code == 200
    finally:
        deleted = client.delete(f"/api/arenas/{arena_id}")
        assert deleted.status_code == 200
