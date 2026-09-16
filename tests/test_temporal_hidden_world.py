from causalrag.benchmarks import compare_temporal_guard, run_temporal_hidden_world


def test_temporal_guard_prevents_stale_immediate_read_for_h1():
    naive, _ = run_temporal_hidden_world("H1", temporal_guard=False)
    guarded, result = run_temporal_hidden_world("H1", temporal_guard=True)

    assert naive.observed_status == "baseline"
    assert naive.success is False
    assert naive.premature_reads == 1
    assert naive.virtual_time_seconds == 0.0

    assert guarded.observed_status == "improved"
    assert guarded.success is True
    assert guarded.premature_reads == 0
    assert guarded.wait_actions == 1
    assert guarded.virtual_time_seconds == 5.0
    assert result.state.pending_effects[0].matched_prediction is True


def test_temporal_guard_turns_delayed_non_response_into_evidence_for_h2():
    naive, _ = run_temporal_hidden_world("H2", temporal_guard=False)
    guarded, result = run_temporal_hidden_world("H2", temporal_guard=True)

    assert naive.conclusion == "unknown"
    assert naive.success is False

    assert guarded.observed_status == "unchanged"
    assert guarded.conclusion == "H2"
    assert guarded.success is True
    assert guarded.wait_actions == 1
    assert guarded.premature_reads == 0

    h1 = result.world_model.get_hypothesis("H1")
    h2 = result.world_model.get_hypothesis("H2")
    assert h1.probability < h2.probability


def test_temporal_guard_comparison_reports_both_hidden_mechanisms():
    report = compare_temporal_guard()

    assert set(report) == {"H1", "H2"}
    assert report["H1"]["naive"]["success"] is False
    assert report["H1"]["temporal_guard"]["success"] is True
    assert report["H2"]["naive"]["success"] is False
    assert report["H2"]["temporal_guard"]["success"] is True
