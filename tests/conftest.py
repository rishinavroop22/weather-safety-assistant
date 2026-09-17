"""Shared fixtures for policy tests."""

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from app.policy import PolicySet, Vocabulary, load_policy

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "policy"

# Benign weather for every fact the shipped SOPs reference: no SOP threshold is crossed.
CALM_WEATHER: dict[str, Any] = {
    "wx.max.temperature_2m": 27.0,
    "wx.max.apparent_temperature": 28.0,
    "wx.max.relative_humidity_2m": 60.0,
    "wx.max.precipitation": 0.0,
    "wx.sum.precipitation": 0.0,
    "wx.max.precipitation_probability": 10.0,
    "wx.max.wind_speed_10m": 8.0,
    "wx.max.wind_gusts_10m": 15.0,
    "wx.max.uv_index": 5.0,
    "wx.min.visibility": 20000.0,
    "wx.codes": [1, 2],
}


@pytest.fixture(scope="session")
def policy() -> PolicySet:
    return load_policy(POLICY_DIR)


@pytest.fixture(scope="session")
def vocabulary(policy: PolicySet) -> Vocabulary:
    return policy.vocabulary


@pytest.fixture
def weather() -> Callable[..., dict[str, Any]]:
    """Factory: calm weather with specific facts overridden."""

    def make(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        return {**CALM_WEATHER, **(overrides or {})}

    return make


def minimal_sop(sop_id: str = "SOP-TEST-01", **fields: Any) -> dict[str, Any]:
    """A valid SOP as a plain dict; override any field via keyword arguments."""
    sop = {
        "id": sop_id,
        "version": 1,
        "title": "Test SOP",
        "category": "test",
        "severity": "moderate",
        "scope": "activity",
        "applies_to": {"activities": ["running"], "groups": []},
        "fallback": False,
        "when": {"all": [{"fact": "wx.max.uv_index", "op": "gte", "value": 7}]},
        "guidance": ["Test guidance."],
        "cite": ["wx.max.uv_index"],
    }
    sop.update(fields)
    return sop


@pytest.fixture
def policy_dir(tmp_path: Path) -> Callable[..., Path]:
    """Factory: a temp policy folder with the real vocabulary and the given SOP files.

    Pass dicts (dumped to ``<id>.yaml``) or ``(file_name, raw_text)`` tuples.
    Pass ``include_shipped=True`` to copy the 12 real SOPs as well.
    """

    def make(*sops: dict[str, Any] | tuple[str, str], include_shipped: bool = False) -> Path:
        root = tmp_path / "policy"
        (root / "sops").mkdir(parents=True, exist_ok=True)
        shutil.copy(POLICY_DIR / "vocabulary.yaml", root / "vocabulary.yaml")
        if include_shipped:
            for path in (POLICY_DIR / "sops").glob("*.yaml"):
                shutil.copy(path, root / "sops" / path.name)
        for sop in sops:
            if isinstance(sop, tuple):
                name, text = sop
                (root / "sops" / name).write_text(text, encoding="utf-8")
            else:
                write_sop(root, sop)
        return root

    return make


def write_sop(root: Path, sop: dict[str, Any], file_name: str | None = None) -> Path:
    path = root / "sops" / (file_name or f"{sop['id']}.yaml")
    path.write_text(yaml.safe_dump(sop, sort_keys=False), encoding="utf-8")
    return path
