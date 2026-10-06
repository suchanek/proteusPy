# Release Notes -- v0.100.6

> Released: 2026-10-06

A dependency-only security release. Four packages flagged by Dependabot now
have declared floors in `pyproject.toml`. Only `urllib3` affects a plain
`pip install proteusPy`; the other three live in optional Poetry groups.

## What changed

- **Dependency floors raised for four Dependabot alerts** (#58). The fix is
  declared in `pyproject.toml`, not just re-locked, so it survives the next
  `poetry lock`:
  - `urllib3>=2.8.0`, now a declared main dependency (it was transitive via
    `requests`): CVE-2026-97687, CVE-2026-97689, CVE-2026-97688.
  - `virtualenv>=21.7.11` in the dev group (transitive via `pre-commit`):
    CVE-2026-102930, CVE-2026-102937, CVE-2026-102925, CVE-2026-102938.
    The lock resolves to 21.14.5, which moves `python-discovery` to 1.6.1.
  - `jupyterlab>=4.6.4` in the jupyter group (transitive via `jupyter`):
    CVE-2026-102831, CVE-2026-102904, CVE-2026-102830.
  - `notebook` floor raised from `>=7.5.6` to `>=7.6.3`: CVE-2026-102831.

  Nothing in proteusPy's own code changes. 289 tests pass.

## Upgrading

Run `pip install -U proteusPy`. It pulls in `urllib3` 2.8.0 or newer. No code
changes are needed.

---

_Full changelog: [CHANGELOG.md](CHANGELOG.md)_
