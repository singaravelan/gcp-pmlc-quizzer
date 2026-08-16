# Testing Policy

All code changes must pass tests before they are considered complete.

## Required checks for every change

Run from repository root:

```bash
.venv/bin/python -m pytest
```

## Minimum quality gate

1. Smoke tests must pass.
2. Regression tests must pass.
3. If a bug fix is made, add or update a regression test that proves the bug is fixed.
4. Do not merge or ship changes while tests are failing.

## Test locations

- `tests/smoke/` for startup/import/basic behavior checks.
- `tests/regression/` for previously fixed bug coverage.
