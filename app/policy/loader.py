"""Load and validate policy files from disk.

Layout::

    policy/
      vocabulary.yaml
      sops/
        SOP-XXX-01.yaml     one SOP per file; the file name must equal the SOP id

``load_policy`` is strict: it validates every file and raises a single
``PolicyError`` listing every problem found. ``PolicyStore`` wraps it for a
long-running server: it reloads when files change and keeps serving the last
valid policy if an edit breaks something.
"""

import logging
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from app.policy.models import SOP, WEATHER_CODE_VARIABLE, WEATHER_CODES_FACT, Vocabulary, iter_leaves

logger = logging.getLogger(__name__)

VOCABULARY_FILE = "vocabulary.yaml"
SOPS_DIR = "sops"
SOP_SUFFIXES = (".yaml", ".yml")

# Union tags Pydantic adds to error locations; hidden to keep messages readable.
_CONDITION_TAGS = {"leaf", "all", "any", "not", "at_least"}


class PolicyError(Exception):
    """Policy files are missing or invalid. ``problems`` lists every issue found."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("Invalid policy configuration:\n" + "\n".join(f"  - {p}" for p in self.problems))


@dataclass(frozen=True)
class PolicySet:
    """A validated vocabulary plus every SOP, ready for the engine."""

    vocabulary: Vocabulary
    sops: tuple[SOP, ...]

    def severity_rank(self, severity: str) -> int:
        return self.vocabulary.severity_ranks[severity]

    def get(self, sop_id: str) -> SOP | None:
        return next((sop for sop in self.sops if sop.id == sop_id), None)

    @property
    def ids(self) -> list[str]:
        return [sop.id for sop in self.sops]


def load_vocabulary(path: Path) -> Vocabulary:
    """Load and validate ``vocabulary.yaml``. Raises ``PolicyError``."""
    data = _read_yaml_mapping(path)
    try:
        return Vocabulary.model_validate(data)
    except ValidationError as exc:
        raise PolicyError(_format_validation_error(path.name, exc)) from None


def load_sop_file(path: Path, vocabulary: Vocabulary) -> SOP:
    """Load and validate one SOP file. Raises ``PolicyError``."""
    data = _read_yaml_mapping(path)
    try:
        sop = SOP.model_validate(data, context={"vocabulary": vocabulary})
    except ValidationError as exc:
        raise PolicyError(_format_validation_error(path.name, exc)) from None
    if sop.id != path.stem:
        raise PolicyError([f"{path.name}: file name does not match id '{sop.id}' (rename it to {sop.id}{path.suffix})"])
    return sop


def list_sop_files(policy_dir: Path) -> list[Path]:
    """SOP files in ``<policy_dir>/sops``, sorted by name."""
    sops_dir = policy_dir / SOPS_DIR
    if not sops_dir.is_dir():
        raise PolicyError([f"SOP directory not found: {sops_dir}"])
    return sorted(p for p in sops_dir.iterdir() if p.is_file() and p.suffix in SOP_SUFFIXES)


def load_policy(policy_dir: Path | str) -> PolicySet:
    """Load the vocabulary and every SOP, failing with all problems at once.

    Raises:
        PolicyError: if the vocabulary is invalid, there are no SOPs, any SOP
            is invalid, or two files define the same id.
    """
    policy_dir = Path(policy_dir)
    vocabulary = load_vocabulary(policy_dir / VOCABULARY_FILE)
    files = list_sop_files(policy_dir)
    if not files:
        raise PolicyError([f"no SOP files found in {policy_dir / SOPS_DIR}"])

    problems: list[str] = []
    sops: list[SOP] = []
    defined_in: dict[str, str] = {}
    for path in files:
        try:
            sop = load_sop_file(path, vocabulary)
        except PolicyError as exc:
            problems.extend(exc.problems)
            continue
        if sop.id in defined_in:
            problems.append(f"{path.name}: duplicate SOP id '{sop.id}' (already defined in {defined_in[sop.id]})")
            continue
        defined_in[sop.id] = path.name
        sops.append(sop)

    if problems:
        raise PolicyError(problems)
    return PolicySet(vocabulary=vocabulary, sops=tuple(sops))


def required_weather_variables(policy: PolicySet) -> list[str]:
    """Open-Meteo variables referenced by any SOP condition or citation.

    Lets the weather client fetch exactly what the policy needs, so an SOP
    that uses a new (vocabulary-listed) variable needs no code change.
    """
    facts: set[str] = set()
    for sop in policy.sops:
        facts.update(leaf.fact for leaf in iter_leaves(sop.when))
        facts.update(sop.cite)

    variables: set[str] = set()
    for fact in facts:
        if fact == WEATHER_CODES_FACT:
            variables.add(WEATHER_CODE_VARIABLE)
        elif fact.startswith("wx."):
            variables.add(fact.split(".")[2])
    return sorted(variables)


class PolicyStore:
    """Thread-safe, reload-on-change access to the policy for a running server.

    * First load failure raises ``PolicyError`` (the app must not start with bad policy).
    * A later broken edit is logged, exposed as ``last_error``, and the last valid
      policy keeps being served until the files are fixed.
    """

    def __init__(self, policy_dir: Path | str) -> None:
        self.policy_dir = Path(policy_dir)
        self.last_error: PolicyError | None = None
        self._policy: PolicySet | None = None
        self._fingerprint: tuple[Any, ...] | None = None
        self._lock = threading.Lock()

    def get(self) -> PolicySet:
        """Return the current policy, reloading if any policy file changed."""
        with self._lock:
            fingerprint = self._fingerprint_files()
            if self._policy is not None and fingerprint == self._fingerprint:
                return self._policy
            try:
                policy = load_policy(self.policy_dir)
            except PolicyError as exc:
                if self._policy is None:
                    raise
                logger.error("Policy reload failed; keeping last valid policy.\n%s", exc)
                self.last_error = exc
                self._fingerprint = fingerprint  # don't re-parse until files change again
                return self._policy
            if self._policy is not None:
                logger.info("Policy reloaded: %d SOPs", len(policy.sops))
            self._policy, self._fingerprint, self.last_error = policy, fingerprint, None
            return policy

    def _fingerprint_files(self) -> tuple[Any, ...]:
        paths = [self.policy_dir / VOCABULARY_FILE]
        sops_dir = self.policy_dir / SOPS_DIR
        if sops_dir.is_dir():
            paths += sorted(p for p in sops_dir.iterdir() if p.suffix in SOP_SUFFIXES)
        entries = []
        for path in paths:
            try:
                stat = path.stat()
                entries.append((path.name, stat.st_mtime_ns, stat.st_size))
            except OSError:
                entries.append((path.name, None, None))
        return tuple(entries)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyError([f"{path}: cannot read file ({exc.strerror or exc})"]) from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise PolicyError([f"{path.name}: invalid YAML: {exc}"]) from None
    if not isinstance(data, dict):
        raise PolicyError([f"{path.name}: expected a mapping at the top level, got {type(data).__name__}"])
    return data


def _format_validation_error(file_name: str, exc: ValidationError) -> list[str]:
    """Turn a Pydantic error into lines like ``SOP-X.yaml: when.all.0.value: ...``."""
    lines = []
    for error in exc.errors():
        parts: list[str] = []
        for part in map(str, error["loc"]):
            if part == "leaf" or (parts and parts[-1] == part and part in _CONDITION_TAGS):
                continue
            parts.append(part)
        location = ".".join(parts) or "(top level)"
        message = error["msg"].removeprefix("Value error, ")
        lines.append(f"{file_name}: {location}: {message}")
    return lines
