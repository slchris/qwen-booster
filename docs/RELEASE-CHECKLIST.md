# Release checklist

State before a public push. Items marked ✅ are done in this repository;
others are decisions or steps that need the maintainer.

## Done

- ✅ MIT `LICENSE` and `THIRD_PARTY_NOTICES.md` (mlx / mlx-lm / mlx-vlm MIT,
  models and drafts listed separately).
- ✅ Bilingual root README (`README.md`, `README.zh-CN.md`) with status matrix
  and honest known gaps.
- ✅ Experiment log moved to `docs/EXPERIMENTS.zh-CN.md`; serving runbook in
  `docs/SERVING.md`.
- ✅ One-off machine-specific runners moved to `scripts/experiments/`.
- ✅ 252 CPU tests, CI workflow added (`.github/workflows/tests.yml`).
- ✅ Package metadata: core deps pinned, `serve` extras, classifiers.

## Before publishing (maintainer decisions)

1. **Repository size.** `benchmarks/` is ~220 MB across ~6.9k files. Options:
   - Keep it (simplest; evidence ships with code, clone is heavier), or
   - Move it to a sibling `qwen-booster-evidence` repository / release asset
     and keep only summaries + links. `git filter-repo` is required to shrink
     history; all `benchmarks/...` links in docs must then point at the new
     location. This repo keeps the links valid for option 1.
2. **Remote and tag.** Add the GitHub remote, set description/topics, push,
   then `git tag -a v0.1.0 -m "Initial public snapshot"` and publish release
   notes from `CHANGELOG.md`.
3. **Scope words.** Keep the "not a serving framework / not universal speedup"
   statements in the README; they are load-bearing for credibility.
4. **Archived logs.** `benchmarks/**/*.log` contain absolute paths and
   prompt/response text from the machine that produced them. Review before
   publishing new archives; existing ones are experiment records.
5. **Ruff/lint.** A CI lint step is not enabled because no lint config exists
   yet. Optional follow-up: add `[tool.ruff]`, fix findings, extend CI.
6. **Coverage badge** and PyPI publishing are optional; this is a toolkit, not
   a library with a stable public API. If published to PyPI, keep the version
   at 0.x and state that internal module APIs may change with the pinned
   MLX-VLM revision.

## Known content gaps (not blockers)

- Tools/function-calling and session endpoints are not validated end-to-end.
- Sustained soak testing and 200K concurrency are unmeasured.
- The immediate-follow-up race fix is a candidate with unproven causality.
