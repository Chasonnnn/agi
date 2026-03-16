"""
Dispatch training for either the candidate span detector or the action model.
"""

from __future__ import annotations

import argparse
import sys

from train_action import main as train_action_main
from train_candidate import main as train_candidate_main


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Dispatch training stages.")
    parser.add_argument("--stage", choices=["candidate", "action"], default="candidate")
    args, remaining = parser.parse_known_args(argv)

    if args.stage == "candidate":
        train_candidate_main(remaining)
        return
    if args.stage == "action":
        train_action_main(remaining)
        return
    raise SystemExit(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main(sys.argv[1:])

