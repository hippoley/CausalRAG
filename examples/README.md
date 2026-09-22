# CausalRAG examples

Start with an example that matches the question you care about. The deterministic examples do **not** require a model key.

## Start here

| Example | What it proves | Run |
| --- | --- | --- |
| `browser_action_guard_demo.py` | a risky external action can be delayed until session state is verified | `python examples/browser_action_guard_demo.py` |
| `tool_routing_demo.py` | routing can be a bounded runtime decision instead of prompt-only prose | `python examples/tool_routing_demo.py` |
| `incident_triage_demo.py` | the best next observation is the one that can change the intervention | `python examples/incident_triage_demo.py` |
| `hidden_world_demo.py` | competing causal hypotheses can be updated from observations | `python examples/hidden_world_demo.py` |
| `open_world_mismatch_demo.py` | the runtime can admit that the current hypothesis set is wrong | `python examples/open_world_mismatch_demo.py` |
| `temporal_hidden_world_demo.py` | fresh-looking evidence can still be causally premature | `python examples/temporal_hidden_world_demo.py` |

## Portable decisions

Run a decision directly from JSON:

```bash
branchpoint decide examples/browser_action_guard.json --json
branchpoint decide examples/value_of_information.json --json
```

These use the same arbitration path as the API and Decision Lab.

## Bring your own domain

`custom_capability_pack.py` shows the extension boundary for:

- world state;
- tools;
- hypotheses;
- experiment contracts;
- intervention contracts;
- defaults and metrics.

Run it with the workbench:

```bash
branchpoint probe \
  --load-pack examples/custom_capability_pack.py:register_pack \
  --open
```

## Older retrieval-oriented examples

This repository started as causal retrieval experiments. Files such as `basic_usage.py`, `causal_extraction_demo.py`, and `evaluate_pipeline.py` preserve that lineage.

They are useful for understanding where CausalRAG came from, but the **current primary surface is the causal decision runtime**.

## Install

From the repository root:

```bash
pip install -e ".[api]"
```

Then choose one example above. For the fastest visual path, use the [verified Pages replay](https://hippoley.github.io/CausalRAG/).
