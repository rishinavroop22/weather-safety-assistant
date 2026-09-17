"""API settings from environment variables (optionally loaded from ``.env``).

FRONTEND_ORIGIN  comma-separated browser origins allowed by CORS
                 (default: http://localhost:5173, the Vite dev server). "*" is rejected.
FRONTEND_DIST    folder with the built frontend to serve at "/" (default: frontend/dist if it exists).
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRONTEND_ORIGIN = "http://localhost:5173"


class ApiSettingsError(ValueError):
    """An API setting is invalid."""


@dataclass(frozen=True)
class ApiSettings:
    frontend_origins: tuple[str, ...]
    frontend_dist: Path | None = None

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, *, load_dotenv_file: bool = True) -> "ApiSettings":
        if env is None:
            if load_dotenv_file:
                from dotenv import load_dotenv

                load_dotenv(override=False)
            env = os.environ

        raw = env.get("FRONTEND_ORIGIN", "").strip() or DEFAULT_FRONTEND_ORIGIN
        origins = tuple(dict.fromkeys(origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()))
        for origin in origins:
            if origin == "*":
                raise ApiSettingsError("FRONTEND_ORIGIN must list explicit origins, not '*'")
            if not origin.startswith(("http://", "https://")):
                raise ApiSettingsError(f"FRONTEND_ORIGIN entries must start with http:// or https:// (got {origin!r})")

        dist_value = env.get("FRONTEND_DIST", "").strip()
        dist = Path(dist_value) if dist_value else ROOT / "frontend" / "dist"
        return cls(frontend_origins=origins, frontend_dist=dist if (dist / "index.html").is_file() else None)
