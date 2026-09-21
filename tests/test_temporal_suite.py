from branchpoint.benchmarks import compare_temporal_suite, run_temporal_suite


def test_temporal_suite_isolates_early_late_and_stacking_failures():
    naive = run_temporal_suite(False)
    by_id = {episode.scenario_id: episode for episode in naive.episodes}

    assert set(by_id) == {
        "early_h1",
        "early_h2",
        "late_window",
        "stacked_intervention",
    }
    assert by_id["early_h1"].premature_reads == 1
    assert by_id["early_h2"].premature_reads == 1
    assert by_id["late_window"].missed_windows == 1
    assert by_id["stacked_intervention"].contaminated_reads == 1


def test_temporal_guard_resolves_each_failure_mode_without_hiding_wait_cost():
    guarded = run_temporal_suite(True)
    by_id = {episode.scenario_id: episode for episode in guarded.episodes}

    assert all(episode.success for episode in guarded.episodes)
    assert sum(episode.premature_reads for episode in guarded.episodes) == 0
    assert sum(episode.missed_windows for episode in guarded.episodes) == 0
    assert sum(episode.contaminated_reads for episode in guarded.episodes) == 0
    assert by_id["late_window"].wait_seconds == 5.0
    assert by_id["stacked_intervention"].wait_seconds == 5.0
    assert guarded.mean_execution_regret > 0.0


def test_temporal_suite_comparison_reports_execution_tradeoff():
    report = compare_temporal_suite()

    naive = report["naive"]
    guarded = report["temporal_guard"]

    assert naive["success_rate"] < guarded["success_rate"]
    assert naive["mean_execution_regret"] > guarded["mean_execution_regret"]
    assert naive["premature_read_rate"] > guarded["premature_read_rate"]
    assert naive["missed_window_rate"] > guarded["missed_window_rate"]
    assert naive["contaminated_read_rate"] > guarded["contaminated_read_rate"]
    assert guarded["mean_wait_seconds"] > 0.0
