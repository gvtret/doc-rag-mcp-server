# Contributing to doc-rag

Thank you for considering a contribution. This document lists the
practical steps and the few rules that exist.

## License

`doc-rag` is licensed under **MIT License**. By contributing, you agree
that your contribution is licensed under the same terms.

We do **not** require a CLA.

If you are contributing on behalf of an employer, please make sure your
employer is aware that your contribution will be released under the MIT
license.

## Development setup

```bash
gh repo clone gvtret/doc-rag-mcp-server doc-rag
cd doc-rag
bash scripts/bootstrap.sh
```

`bootstrap.sh` installs [uv](https://docs.astral.sh/uv/) if missing,
prompts for extras, and runs `uv sync --frozen` to materialise the
venv into `.venv/`.

System packages you may need for full development (Debian/Ubuntu):

```bash
sudo apt install antiword
```

## Running tests

```bash
pytest
pytest --cov=doc_rag --cov-report=term-missing
```

CI runs on Python 3.10, 3.11, and 3.12. Your patch must pass on all
three.

## Linting and formatting

We use [Ruff](https://github.com/astral-sh/ruff) for both:

```bash
ruff check .
ruff format .
```

CI runs `ruff check` and `ruff format --check`. Run both locally before
opening a PR.

## Commit messages

- Short imperative subject line (≤ 72 characters): `Fix FAISS reconstruct
  for sparse indices`, not `fixed bugs`.
- If the change is non-trivial, add a body explaining **why**, separated
  from the subject by a blank line.
- Reference issues with `Fixes #123` or `Refs #123` when relevant.
- Do not include co-author trailers from automated tools unless the
  user explicitly asks.

## Pull request checklist

Before opening a PR:

1. Your branch is rebased on current `master`.
2. `pytest` passes locally.
3. `ruff check .` and `ruff format --check .` pass.
4. If you added a user-facing change, `CHANGELOG.md` has an entry under
   `## Unreleased`.
5. If you changed the public surface defined in `docs/roadmap.md`
   (CLI, MCP tools, HTTP endpoints, config keys, on-disk schema), the
   PR description says so explicitly and proposes a SemVer impact
   (`patch` / `minor` / `major`).
6. If you added a dependency, `NOTICE` lists it with its license.

## Reviewing scope

PRs that touch unrelated files, reformat code outside the scope of
the change, or rename things "while we're here" will be asked to be
split. Surgical changes are easier to review and easier to revert.

If you find unrelated bugs while working on something, please open an
issue rather than fixing them in the same PR.

## Things we are unlikely to merge

- Adding telemetry or analytics that phone home.
- Optional dependencies on third-party hosted services.
- Backwards-incompatible changes without a SemVer bump and a deprecation
  path (see `docs/roadmap.md` § 1).
- Bundling new fonts, images, or other non-code assets above a few KB
  without discussion.

## Asking questions

Open an issue with the `question` label. For larger design questions,
open a discussion thread before writing code — it saves both sides time.

## Code of conduct

Be civil. The project has too few maintainers to spend time on
unfriendly interactions. We will lock or remove threads that stop being
productive, without long explanations.
