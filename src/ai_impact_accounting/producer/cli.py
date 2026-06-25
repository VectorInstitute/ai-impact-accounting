"""The ``dia`` command-line interface.

A Phase-1 norm-setting lever: drop it into a CI job or a conference submission
check to validate a model card's ``dia_report`` block.

.. code-block:: console

    dia validate README.md          # a local model card
    dia validate you/model          # a Hub repo id
    dia report   you/model          # show the parsed footprint
"""

from __future__ import annotations

import json
import sys
from importlib import resources

import jsonschema
import yaml

from ai_impact_accounting.hub.ingest import fetch_meta


def _load_schema() -> dict:
    """Load the packaged ``dia_schema.json`` resource."""
    with resources.files("ai_impact_accounting.schema").joinpath("dia_schema.json").open() as f:
        return json.load(f)


def cmd_validate(target: str) -> int:
    """Validate a card's ``dia_report`` against the schema.

    Parameters
    ----------
    target : str
        A local card path or a Hub repo id.

    Returns
    -------
    int
        ``0`` if valid, ``1`` otherwise.
    """
    meta = fetch_meta(target)
    block = meta.get("dia_report")
    if block is None:
        print(f"FAIL  no dia_report block in {target}")
        return 1
    schema = _load_schema()
    try:
        jsonschema.validate(block, schema)
    except jsonschema.ValidationError as e:
        print(f"FAIL  {target}: {e.message} (at {'/'.join(map(str, e.path))})")
        return 1
    if block.get("scope") != "incremental":
        print(f"FAIL  {target}: scope must be 'incremental'")
        return 1
    print(f"OK    {target}: valid DIA report")
    return 0


def cmd_report(target: str) -> int:
    """Print the parsed ``dia_report`` block as YAML.

    Parameters
    ----------
    target : str
        A local card path or a Hub repo id.

    Returns
    -------
    int
        Always ``0``.
    """
    meta = fetch_meta(target)
    print(yaml.safe_dump(meta.get("dia_report", {}), sort_keys=False))
    return 0


def main() -> int:
    """Entry point for the ``dia`` console script.

    Returns
    -------
    int
        Process exit code.
    """
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cmd, target = sys.argv[1], sys.argv[2]
    handler = {"validate": cmd_validate, "report": cmd_report}.get(cmd)
    if handler is None:
        print(__doc__)
        return 2
    return handler(target)


if __name__ == "__main__":
    sys.exit(main())
