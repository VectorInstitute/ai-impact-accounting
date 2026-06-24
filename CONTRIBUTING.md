# Contributing to ai-impact-accounting

Thanks for your interest in contributing to ai-impact-accounting!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Pre-commit hooks

Once the python virtual environment is setup, you can run pre-commit hooks using:

```bash
pre-commit run --all-files
```

## Coding guidelines

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).

For docstrings we use [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).

We use [ruff](https://docs.astral.sh/ruff/) for code formatting and static code
analysis. Ruff checks various rules including [flake8](https://docs.astral.sh/ruff/faq/#how-does-ruff-compare-to-flake8). The pre-commit hooks show errors which you need to fix before submitting a PR.

Last but not the least, we use type hints in our code which is then checked using
[mypy](https://mypy.readthedocs.io/en/stable/).

## Secrets and local artifacts

Never commit Hugging Face tokens, API keys, or credential files. Use
``huggingface-cli login`` or ``HF_TOKEN`` in your environment instead.

Training outputs (``out-*/``, checkpoints, ``powermetrics_log.txt``,
``coverage.xml``, ``emissions.csv``) and ``.env`` / ``.huggingface/`` files are
listed in ``.gitignore`` — keep them local. Never commit ``HF_TOKEN`` values or
API keys in source, scripts, or model cards.

## Continuous integration

GitHub Actions runs on pushes and PRs to ``main`` and ``dia-code``:

| Workflow | What it runs |
|---|---|
| [code checks](.github/workflows/code_checks.yml) | ``pre-commit run --all-files`` (ruff, mypy, doctests, unit tests) + ``pip-audit`` |
| [unit tests](.github/workflows/unit_tests.yml) | ``pytest -m "not integration_test"`` with coverage |
| [integration tests](.github/workflows/integration_tests.yml) | ``pytest -m "integration_test"`` (needs ``HF_TOKEN`` repo secret; self-skips if unset) |

Reproduce locally:

```bash
uv sync --all-extras --dev
source .venv/bin/activate
pre-commit run --all-files
uv run pytest -m "not integration_test" --cov src/ai_impact_accounting tests
HF_TOKEN=... uv run pytest -m "integration_test" tests   # optional Hub tests
```
