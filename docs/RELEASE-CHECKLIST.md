# Release checklist

State before a public push. Items marked ✅ are done in this repository;
others are steps that need the maintainer.

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
- ✅ Evidence archives are **not tracked** in the source repository. They are
  distributed as a release bundle; `docs/EVIDENCE.md` explains download,
  verification, local rebuild and local scoring.

## Publishing steps

1. **Build the evidence bundle**

   ```bash
   python scripts/pack_evidence.py --source benchmarks \
     --output dist/qwen-booster-evidence-v0.1.0.tar.gz
   ```

   This writes a deterministic tarball plus a `.sha256` sidecar. Rebuild if
   `benchmarks/` changes before the release.

2. **Add the remote and push**

   ```bash
   git remote add origin <your repository URL>
   git push -u origin master   # or rename the branch to main first
   ```

3. **Tag and publish**

   ```bash
   git tag -a v0.1.0 -m "Initial public snapshot"
   git push origin v0.1.0
   # Create the GitHub release from CHANGELOG.md and attach
   # dist/qwen-booster-evidence-v0.1.0.tar.gz and its .sha256 sidecar.
   ```

4. **Repository settings**: description, topics (`apple-silicon`, `mlx`,
   `llm-inference`, `benchmark`), and enable the CI workflow.

## Notes and optional follow-ups

- **Git history** still contains the initial import of `benchmarks/`. The
  packed `.git` is small (~8 MB) because the files compress well, so a
  history rewrite is optional. If a clean history is required, use
  `git filter-repo --path benchmarks --invert-paths` and re-tag afterwards.
- **Lint**: no ruff configuration exists; a CI lint step would need a
  baseline pass first.
- **Scope words**: keep the "not a serving framework / not universal speedup"
  statements in the README; they are load-bearing for credibility.
- **Archived logs**: `benchmarks/**/*.log` contain absolute paths and
  prompt/response text from the author's machine. Review before publishing a
  bundle built from new archives.
- **PyPI** publishing is optional; internal module APIs track the pinned
  MLX-VLM revision and are not a stable public interface.

## Known content gaps (not blockers)

- Tools/function-calling and session endpoints are not validated end-to-end.
- Sustained soak testing and 200K concurrency are unmeasured.
- The immediate-follow-up race fix is a candidate with unproven causality.
