<!--
proteusPy pull request template.

House style: lead with *why*, then what changed, then what you actually ran.
Evidence beats assertion -- paste the failing command, the traceback, the
measured output. Delete any section that does not apply; an empty heading is
worse than no heading. These comments are invisible in the rendered PR, so
leave them in place if they are still useful to you.
-->

## Why

<!--
The problem this solves, and the evidence for it. Show the reader the defect
rather than asserting it: a failing command and its output, a traceback, a
response code, a resolved URL. If it fixes an issue, "Closes #N".

For a release PR, say why the release exists -- what it corrects that cannot be
corrected in place.
-->

## Changes

<!--
What changed, grouped by file or by concern, in the order a reviewer should read
them. The diff already says what; use this space for why, and for anything a
reviewer would otherwise have to reverse-engineer.
-->

## Test plan

<!--
What you ran, with results. Use an unchecked box for anything you could NOT
verify, and say why -- "not run here, pycodekg isn't installed in this venv" is
useful; silence is not. Add the commands specific to your change.
-->

- [ ] `pytest -k "not test_disulfide_class_generator"` — the selection CI runs — passes: <!-- N passed -->
- [ ] `ruff check proteusPy/` clean
- [ ] `pre-commit run --files <changed files>` clean
- [ ] Doctests still pass for any module whose docstrings changed (they are
      formatting-sensitive — see the Testing section of `README.md`)

## Notes for review

<!--
Optional, and often the most valuable section. Judgement calls you want a second
opinion on; things you deliberately did NOT do, and why; findings that are a
maintainer's call rather than yours; anything that looks wrong in the diff but
is intentional.
-->

## Checklist

- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] No large or binary files added. The big `.pkl` files are GitHub Release
      assets fetched by `proteusPy/data_fetch.py`, not repository content — see
      the Data Files section of `README.md`. `check-added-large-files` rejects
      anything over 1 MB, so if you are tempted to bypass it, say why here.
- [ ] Version untouched, **or** bumped across all five surfaces that carry it:
      `pyproject.toml`, `proteusPy/_version.py`, `CITATION.cff`, the `README.md`
      BibTeX block, and `CHANGELOG.md` / `release-notes.md`
