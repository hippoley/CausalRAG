# Release readiness

Branchpoint is built and tested as an installable Python distribution before any public package release.

## Artifact gate

Every packaging change must pass `.github/workflows/package-ci.yml`.

The gate:

1. builds both wheel and source distribution;
2. runs `twine check` on both artifacts;
3. verifies that the wheel contains the execution, authorization, Jev bridge, and browser templates;
4. installs the wheel into a clean virtual environment outside the repository;
5. runs the installed `branchpoint` CLI;
6. exercises authorization, downstream idempotency propagation, durable receipts, and replay from the installed wheel.

This catches a class of failures that editable installs cannot: missing package data, stale entry points, metadata errors, or imports that accidentally depend on the repository checkout.

## Public publishing

The repository does not claim a PyPI release until a package is actually published and independently installable from the index.

Before the first public upload:

- confirm the final distribution name on PyPI;
- configure a trusted publisher or another minimal-scope publishing credential;
- build from a tagged commit whose CI is green;
- verify the tag version matches `branchpoint.__version__`;
- publish immutable artifacts;
- install the published wheel in a clean environment and repeat the artifact smoke test;
- create a GitHub release containing the same version and checksums.

Do not make publishing credentials available to ordinary pull-request jobs.

## Version contract

The current package version is `0.3.0`.

The installed CLI must report:

```text
Branchpoint 0.3.0
```

The repository URL may remain `hippoley/CausalRAG` for compatibility; that does not change the public runtime/package identity.
