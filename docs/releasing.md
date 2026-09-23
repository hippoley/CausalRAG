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

This catches failures that editable installs cannot: missing package data, stale entry points, metadata errors, or imports that accidentally depend on the repository checkout.

## One-time PyPI setup

Public publishing uses PyPI Trusted Publishing. Do not store a long-lived PyPI token in this repository.

Configure a Trusted Publisher for the `branchpoint` project, or a pending publisher before the first release, with:

- GitHub owner: `hippoley`
- Repository: `CausalRAG`
- Workflow: `release.yml`
- Environment: `pypi`

Create a GitHub environment named `pypi`. Environment protection rules are recommended so a release can require an explicit approval without exposing a package credential.

Confirm that the final distribution name is available or already controlled by the project before creating the first release tag.

## Release contract

1. Update `branchpoint.__version__` and version-specific tests in the release PR.
2. Merge only after normal CI and `package-ci` are green.
3. Create a tag matching the package version exactly, for example `v0.3.0`.
4. The tag-triggered `release.yml` workflow:
   - builds wheel and sdist;
   - runs `twine check`;
   - installs the wheel in a clean environment;
   - rejects a tag/version mismatch;
   - creates GitHub build-provenance attestations;
   - publishes the same distributions to PyPI through OIDC Trusted Publishing;
   - creates a GitHub Release with those distributions and `SHA256SUMS`.
5. Verify the PyPI release and GitHub Release before announcing the version.

A manually dispatched release workflow only builds and validates artifacts. Publishing is restricted to tag refs.

## Recovery rule

Published package versions are immutable. If a release partially succeeds, inspect PyPI and GitHub first, fix the workflow, bump the package version, and publish a new release. Do not overwrite an existing release artifact.

## Version contract

The current package version is `0.3.0`.

The installed CLI must report:

```text
Branchpoint 0.3.0
```

The repository URL may remain `hippoley/CausalRAG` for compatibility; that does not change the public runtime/package identity.
