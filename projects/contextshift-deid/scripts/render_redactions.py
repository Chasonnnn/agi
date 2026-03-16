from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contextshift_deid.constants import ARTIFACTS_DIR, INTERIM_DIR
from contextshift_deid.data import load_jsonl
from contextshift_deid.redaction import RedactionRenderer, RedactionSpan


def main() -> None:
    parser = argparse.ArgumentParser(description="Render session transcripts with deterministic redaction strategies.")
    parser.add_argument(
        "--input",
        type=Path,
        default=INTERIM_DIR / "upchieve_math_sessions.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ARTIFACTS_DIR / "redaction_preview.jsonl",
    )
    parser.add_argument(
        "--strategy",
        choices=["typed_placeholder", "mask", "delete", "fake_surrogate"],
        default="typed_placeholder",
    )
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()

    sessions = load_jsonl(args.input)
    rows: list[dict] = []
    for session in sessions[: args.limit]:
        renderer = RedactionRenderer(strategy=args.strategy)
        rendered_turns = []
        for turn in session["turns"]:
            spans = [
                RedactionSpan(
                    start=int(annotation["start"]),
                    end=int(annotation["end"]),
                    text=str(annotation["text"]),
                    entity_type=str(annotation["pii_type"]),
                )
                for annotation in turn.get("annotations", [])
            ]
            rendered_turns.append(
                {
                    "role": turn["role"],
                    "original": turn["content"],
                    "rendered": renderer.render(turn["content"], spans),
                    "annotations": turn.get("annotations", []),
                }
            )
        rows.append(
            {
                "session_id": session["session_id"],
                "subject": session["subject"],
                "strategy": args.strategy,
                "turns": rendered_turns,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"Wrote {len(rows)} rendered sessions to {args.output}")


if __name__ == "__main__":
    main()
