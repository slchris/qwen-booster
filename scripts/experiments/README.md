# scripts/experiments

Historical one-off runners from the original investigation (`run_*`,
`review_*`, experiment-specific `evaluate_*`, kernel screens, probes and
traces). They contain absolute paths from the author's machine and are kept
for provenance; they are not supported tooling.

They cross-import each other and the product evaluators by module name, so if
you need to run one, set both directories on the path:

```bash
PYTHONPATH=src:scripts:scripts/experiments \
  .venv/bin/python scripts/experiments/evaluate_200k.py --help
```

For supported commands see `../README.md` and the root `README.md`.
