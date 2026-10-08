# Contributing

Thanks for looking at qwen-booster. This project is evidence-first: a change
that affects performance, correctness or memory is only accepted together with
the measurement that supports it.

## Development setup

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.lock
uv pip install --no-deps -e .
python -m pytest -q
```

The test suite is CPU-only and runs in a few seconds. Real-model checks need
Apple Silicon, the local model snapshots, and must run one GPU experiment at a
time.

## What we expect in a pull request

1. **Keep failures.** Do not delete or rewrite archived results. New evidence
   goes into a new `benchmarks/<name>/` directory via
   `scripts/archive_run.py`, including raw outputs and a manifest.
2. **State the scope.** Say what was measured (model, quantization, context
   length, cache state, repetitions) and what is *not* covered. Single
   observations are allowed but must be labelled as such.
3. **No extrapolation.** Short-context gains are not long-context claims;
   microbenchmarks are not end-to-end results. Use the wording the existing
   READMEs use.
4. **Fail closed.** Cache and admission changes must preserve correctness when
   state cannot be verified, rather than guessing.
5. **Tests.** Add CPU tests for new logic. GPU runs are documented in the
   archive, not in CI.

## Reporting bugs

Open an issue with the command you ran, the archive/manifest if available, and
the observed vs expected behaviour. For security issues see `SECURITY.md`.
