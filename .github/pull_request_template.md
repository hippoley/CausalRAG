## What changed?

<!-- One behavior change. Keep this narrow. -->

## Why?

<!-- What failure mode, decision boundary, or user friction does this address? -->

## Reproduction / proof

<!-- Prefer a test, trace, same-world comparison, or screenshot over a prose-only claim. -->

```text
before:
after:
```

## Truth boundary

- [ ] I am not presenting a frontend replay as live backend execution.
- [ ] Any benchmark number in this PR is reproducible from repository code or artifacts.
- [ ] Any external claim is linked to evidence.
- [ ] Human overrides still respect runtime validity.

## Verification

- [ ] `pytest -q`
- [ ] `python -m compileall branchpoint`
- [ ] I ran the affected example or surface manually.
- [ ] UI changes include a screenshot or short capture when useful.

## Scope

- [ ] No unrelated refactor.
- [ ] No secrets, private data, or proprietary datasets.
