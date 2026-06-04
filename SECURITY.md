# Security Policy

## Supported versions

`doc-rag` follows Semantic Versioning. Security fixes are applied to the
latest released minor version on the `main` branch. Older minor versions
do **not** receive backports.

| Version | Status |
| ------- | ------ |
| 1.x (latest minor) | Security fixes |
| < 1.x | Unsupported |

If you are running an older release, the recommended remediation is to
upgrade to the latest 1.x release before reporting an issue you cannot
otherwise demonstrate against `main`.

## Reporting a vulnerability

Please **do not** open a public GitHub issue for security problems.

Email reports to: `yas.tretel@gmail.com`

Include:

- a short description of the issue and its impact,
- the commit SHA or release tag you reproduced against,
- minimal reproduction steps (a config snippet, a curl command, a
  malformed document, etc.),
- whether you would like to be credited in the eventual advisory.

If you would like to send the report encrypted, ask for a current PGP key
first in a short plaintext email.

## Response timeline

This is a small project with a single primary maintainer. Realistic
targets:

| Step | Target |
| ---- | ------ |
| Acknowledgement of receipt | within 7 days |
| Initial triage and severity assessment | within 14 days |
| Fix or mitigation in `main` | depends on severity; critical issues prioritised |
| Public advisory | after a fix is available and users have had a reasonable upgrade window |

If you do not receive an acknowledgement within 7 days, please follow up
on the same thread — the original email may have been missed.

## Scope

In scope:

- the `doc-rag` Python package and its CLI;
- the bundled HTTP/MCP server (`src/doc_rag/server/`);
- the bundled Web UI (templates and routes under `/ui/*`);
- the bundled Docker image (`docker/Dockerfile`) and native installer
  (`scripts/install_server_native.sh`).

Out of scope:

- vulnerabilities in third-party dependencies, unless we ship a vendored
  copy or our default config materially amplifies the impact (we will
  still help you report upstream);
- attacks that require attacker-controlled local file write inside the
  data root (`build/`, `sources/`) — the trust boundary is the network,
  not the local filesystem;
- DoS through extremely large source documents — set `parsing.max_file_mb`
  and related limits in `config/config.yaml` to protect production
  deployments.

## Hardening defaults users should know about

- The MCP/UI server is **unauthenticated by default**. Set
  `DOC_RAG_API_KEY` to enable bearer-token auth before exposing the
  server outside a trusted LAN.
- `DOC_RAG_ALLOWED_ORIGINS` should be an explicit allow-list, not `*`,
  for any deployment that receives requests with `Origin` headers.
- The Web UI exposes destructive operations (`/ui/delete`, `/ui/wipe`).
  Treat the API key (if set) as a destructive credential, not just a
  read token.

## Credit

Reporters who follow this policy will be credited in the published
advisory and in `CHANGELOG.md` unless they prefer to remain anonymous.
