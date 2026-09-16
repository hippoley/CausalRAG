# Changelog

## 0.3.0 — 2026-09-16

CausalRAG v0.3 moves the project from causal retrieval toward a causal decision runtime with explicit uncertainty, experiments, interventions, and behavioral evaluation.

### Added

- Explicit competing hypotheses in `CausalWorldModel` with falsification-oriented reasoning.
- Runtime-owned hypothesis discrimination so model self-reported information gain is not authoritative.
- `ExperimentContract` for discrete `P(outcome | hypothesis, action)` likelihood models.
- Bayesian expected information gain and exact posterior updates from experiment outcomes.
- Protection against double-counting Bayesian evidence through a second LLM hypothesis update.
- `InterventionContract` for expected consequence utility under competing hypotheses.
- One-step expected value of sample information (EVSI) for comparing another observation with acting now.
- `DecisionPreferences` as a deployment-owned runtime surface for intervention consequence utilities.
- Decision traces exposing Bayesian information gain, decision-value source, EVSI, and net sampling value.
- HiddenWorld deterministic and seeded stochastic causal-agency benchmarks.
- Behavioral metrics including success, mechanism identification, posterior calibration/Brier score, probe count, total capability cost, and causal regret.
- Fixed-world policy comparison harness for greedy-EIG, conservative-EIG, decision-value, random-probe, and cheapest-probe policies.
- Utility-sensitivity sweeps that hold the environment and seeds fixed while varying wrong-action loss.
- No-key demos for falsification, Bayesian experiments, decision preferences, HiddenWorld, policy comparison, and utility sensitivity.

### Changed

- Retrieval is treated as one optional evidence capability instead of the core agent ontology.
- `ToolSpec` cost, risk, and reversibility are enforced by the runtime for every proposer type.
- The LLM/reasoner is treated as a candidate proposer rather than the final action authority.
- Core installation remains lightweight; local neural embeddings and FAISS stay opt-in.

### Fixed

- Bayesian evidence direction is evaluated relative to normalized prior probability rather than raw unnormalized hypothesis credence.
- Runtime Bayesian updates bypass a second model-driven hypothesis update for the same observation.
- Custom reasoners can no longer win action selection by under-reporting capability cost, risk, or irreversibility.

### Benchmark notes

On the repository's fixed stochastic HiddenWorld suite of 30 episodes (three hidden mechanisms × seeds 0 through 9), the neutral one-step decision-value policy measured 0.833 success and 0.251 mean causal regret. A utility-sensitivity sweep showed a policy phase change when wrong-intervention loss moved between 0.5 and 1.0; at loss 1.0, the same fixed suite measured 0.900 success and 0.216 mean causal regret.

These numbers describe that repository benchmark only. They are not general claims about performance on external tasks.

### Compatibility

- The legacy `CausalRAGPipeline` and one-shot RAG CLI path remain available.
- Python 3.10+ is supported by the v0.3 runtime; core CI covers Python 3.10 and 3.12.
