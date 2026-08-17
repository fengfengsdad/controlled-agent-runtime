from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from agent_runtime.config import settings


def catalog_path() -> Path:
    return settings.data_dir / "requirement_catalog.json"


def ensure_default_catalog(path: Optional[Path] = None) -> Path:
    target = path or catalog_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_text(
            json.dumps(
                {
                    "CHG-1001": {
                        "title": "Add idempotent webhook retries",
                        "owner": "payments-platform",
                        "status": "draft",
                        "constraints": [
                            "No duplicate side effects",
                            "Emit audit event on write",
                        ],
                    },
                    "CHG-1002": {
                        "title": "Introduce approval gate for agent writes",
                        "owner": "ai-platform",
                        "status": "ready",
                        "constraints": [
                            "Human approval before mutating tools",
                            "Prompt-injection rejection enabled",
                        ],
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return target


def load_catalog(path: Optional[Path] = None) -> Dict[str, Any]:
    target = ensure_default_catalog(path)
    return json.loads(target.read_text(encoding="utf-8"))


def lookup_requirement(change_id: str, path: Optional[Path] = None) -> Dict[str, Any]:
    catalog = load_catalog(path)
    item = catalog.get(change_id)
    if item is None:
        return {
            "ok": False,
            "error": f"change_id not found: {change_id}",
            "data": {},
        }
    return {
        "ok": True,
        "error": None,
        "data": {"change_id": change_id, **item},
    }
