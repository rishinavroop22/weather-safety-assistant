"""Validate the policy folder from the command line.

    python -m app.policy            # validates ./policy
    python -m app.policy path/to/policy

Exit code 0 = valid, 1 = invalid. Handy right after adding a new SOP file.
"""

import sys
from pathlib import Path

from app.policy.loader import PolicyError, load_policy, required_weather_variables


def main(argv: list[str]) -> int:
    policy_dir = Path(argv[1]) if len(argv) > 1 else Path("policy")
    try:
        policy = load_policy(policy_dir)
    except PolicyError as exc:
        print(exc)
        return 1
    print(f"OK: {len(policy.sops)} SOPs loaded from {policy_dir}")
    for sop in sorted(policy.sops, key=lambda s: (-policy.severity_rank(s.severity), s.id)):
        flag = " (fallback)" if sop.fallback else ""
        print(f"  {sop.id:<13} {sop.severity:<9} {sop.scope:<8} {sop.title}{flag}")
    print("Weather variables needed:", ", ".join(required_weather_variables(policy)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
