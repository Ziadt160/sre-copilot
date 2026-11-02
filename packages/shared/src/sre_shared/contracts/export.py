"""Write the chat API's JSON Schemas for the web UI: `python -m sre_shared.contracts.export`.

The file is committed; a test fails when it is stale, so the UI's generated types never drift
from `contracts/agent.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from sre_shared.contracts.agent import json_schemas
from sre_shared.settings import REPO_ROOT

SCHEMA_PATH = REPO_ROOT / "web" / "src" / "contracts" / "agent.schema.json"


def render() -> str:
    return json.dumps(json_schemas(), indent=2, sort_keys=True) + "\n"


def main(path: Path = SCHEMA_PATH) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render(), encoding="utf-8", newline="\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
