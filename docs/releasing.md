# Release readiness

Branchpoint is built and tested as an installable Python distribution before any
public package release.

## Release identity

| Field | Value |
| --- | --- |
| PyPI project / distribution | `branchpoint` |
| Python import package | `branchpoint` |
| GitHub owner | `hippoley` |
| GitHub repository | `CausalRAG` |
| Trusted Publisher workflow | `release.yml` |
| GitHub environment | `pypi` |
| Release tag shape | `v<package-version>` |

The repository URL remains `hippoley/CausalRAG` for link compatibility even
though the public package identity is Branchpoint.

## Artifact gate

Every packaging change must pass `.github/workflows/package-ci.yml`.

The gate:

1. builds both wheel and source distribution;
2. runs `twine check` on both artifacts;
3. runs `scripts/verify_distribution.py` against the wheel;
4. installs the wheel into a clean virtual environment outside the repository;
5. runs the installed `branchpoint` CLI;
6. exercises authorization, downstream idempotency propagation, durable
   receipts, and replay from the installed wheel.

The distribution verifier checks the package name/version, required runtime
files, the PostgreSQL execution backend, the OpenAI Agents integration module,
advertised extras, and the dependency linkage for `postgres` and
`openai-agents`.

This catches failures that editable installs cannot: missing package data, stale
entry points, metadata errors, or imports that accidentally depend on the
repository checkout.

## One-time PyPI setup

Public publishing uses PyPI Trusted Publishing. Do not store a long-lived PyPI
token in this repository.

If the `branchpoint` project does not yet exist, create a **pending GitHub
Trusted Publisher** with:

- PyPI project name: `branchpoint`
- GitHub owner: `hippoley`
- Repository: `CausalRAG`
- Workflow: `release.yml`
- Environment: `pypi`

A pending publisher does not reserve the project name. Configure it only when
the repository is ready to release and publish the first version promptly.

Create a GitHub environment named `pypi`. Environment protection rules are
recommended so a release can require explicit approval without exposing a
package credential. The PyPI publisher must use the same environment name.

The repository currently uses the PyPA publish action with
`id-token: write`; no username, password, or stored PyPI API token is required.

## Before tagging

All of the following should be true:

1. `main` CI is green.
2. `package-ci` is green for Python 3.10 and 3.12.
3. A fresh local build passes `twine check` and
   `scripts/verify_distribution.py`.
4. `branchpoint.__version__` is the intended release version.
5. The tag points to a commit reachable from `origin/main`.
6. Public docs do not claim a PyPI install path until the first upload exists.

Optional local preflight:

```bash
python -m pip install --upgrade build twine
rm -rf dist build
python -m build
python -m twine check dist/*
python scripts/verify_distribution.py dist
```

## Release contract

For version `0.3.0`:

```bash
git checkout main
git pull --ff-only
git tag -a v0.3.0 -m "Branchpoint 0.3.0"
git push origin v0.3.0
```

The tag-triggered `.github/workflows/release.yml` then executes:

```text
build distributions
→ twine check
→ verify_distribution.py
→ verify tag commit is on main
→ verify tag == package version
→ clean-wheel install smoke
→ upload GitHub artifact
→ artifact attestation
→ PyPI Trusted Publishing
→ SHA256SUMS
→ GitHub Release
```

The GitHub Release is created only after PyPI publishing succeeds.

The workflow refuses a release tag whose commit is not an ancestor of
`origin/main`, and refuses a tag whose version does not exactly match
`branchpoint.__version__`.

## Dry run without publishing

The `release` workflow supports manual dispatch. Manual dispatch runs build,
distribution verification, clean-wheel smoke, and artifact upload, but skips
the tag-only attestation, PyPI publishing, and GitHub Release jobs.

Use that dry run before the first public package upload.

## Recovery rule

Published package versions are immutable. If a release partially succeeds,
inspect PyPI and GitHub first, fix the workflow, bump the package version, and
publish a new release. Do not overwrite an existing release artifact.

## After the first successful PyPI release

Only after PyPI shows the real `branchpoint` release should public install
instructions change from source installation to:

```bash
pip install branchpoint
```

Optional integrations then become:

```bash
pip install "branchpoint[openai-agents]"
pip install "branchpoint[postgres]"
```

Until that release exists, keep the public documentation source-grounded.

## Version contract

The current package version is `0.3.0`.

The installed CLI must report:

```text
Branchpoint 0.3.0
```
