# Security policy

## Scope

qwen-booster is a local, single-user experiment tool. The serving paths bind
`127.0.0.1` by default. Binding to a LAN address (`--host`) and enabling
`--api-key` are explicit opt-ins and are not hardened for hostile networks.

## Defaults

- No automatic downloads: model and draft loading only reads local paths.
- No telemetry: nothing is sent anywhere by this project.
- `--api-key` uses a plain bearer token; use a reverse proxy with TLS for any
  real deployment.
- Archived server logs under `benchmarks/` may contain local paths and
  prompt/response text from experiments. Review before publishing new
  archives.

## Reporting a vulnerability

Please use GitHub private security advisories for this repository ("Security"
tab -> "Report a vulnerability"). Do not open a public issue for an
exploitable problem. Include the affected command, configuration, and a
minimal reproduction if possible.
