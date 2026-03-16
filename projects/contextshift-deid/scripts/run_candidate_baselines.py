from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import spacy
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def tokenize_with_spans(text: str) -> tuple[list[str], list[tuple[int, int]]]:
    tokens: list[str] = []
    spans: list[tuple[int, int]] = []
    for match in TOKEN_RE.finditer(text):
        tokens.append(match.group(0))
        spans.append(match.span())
    return tokens, spans


def merge_char_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def build_bio_labels(token_spans: list[tuple[int, int]], char_spans: list[tuple[int, int]]) -> list[str]:
    labels = ["O"] * len(token_spans)
    for start, end in merge_char_spans(char_spans):
        first = True
        for index, (token_start, token_end) in enumerate(token_spans):
            overlaps = token_start < end and token_end > start
            if not overlaps:
                continue
            labels[index] = "B-SUSPECT" if first else "I-SUSPECT"
            first = False
    return labels


def build_text_lookup(interim_rows: list[dict]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for session in interim_rows:
        session_id = str(session["session_id"])
        for turn in session["turns"]:
            lookup[f"{session_id}-turn-{turn['sequence_id']}"] = str(turn["content"])
    return lookup


def run_spacy(nlp, text: str) -> list[tuple[int, int]]:
    doc = nlp(text)
    return [(ent.start_char, ent.end_char) for ent in doc.ents]


def run_presidio(analyzer: AnalyzerEngine, text: str, score_threshold: float | None) -> list[tuple[int, int]]:
    kwargs = {"text": text, "language": "en"}
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    results = analyzer.analyze(**kwargs)
    return [(result.start, result.end) for result in results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run spaCy and Presidio candidate baselines.")
    parser.add_argument("--split-file", type=Path, default=Path("data/processed/candidate/dev.jsonl"))
    parser.add_argument("--interim-file", type=Path, default=Path("data/interim/upchieve_math_sessions.jsonl"))
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--spacy-output", type=Path, default=Path("artifacts/predictions/candidate_dev_spacy.jsonl"))
    parser.add_argument("--presidio-output", type=Path, default=Path("artifacts/predictions/candidate_dev_presidio.jsonl"))
    parser.add_argument("--presidio-score-threshold", type=float, default=None)
    args = parser.parse_args()

    split_rows = load_jsonl(args.split_file)
    interim_rows = load_jsonl(args.interim_file)
    text_lookup = build_text_lookup(interim_rows)

    nlp = spacy.load(args.spacy_model)
    nlp_configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": args.spacy_model}],
    }
    provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
    analyzer = AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])

    spacy_predictions: list[dict] = []
    presidio_predictions: list[dict] = []
    mismatches = 0

    for row in split_rows:
        record_id = row["id"]
        text = text_lookup.get(record_id)
        if text is None:
            raise SystemExit(f"Missing raw text for {record_id}")

        tokens, token_spans = tokenize_with_spans(text)
        if tokens != row["tokens"]:
            mismatches += 1
            raise SystemExit(
                f"Token mismatch for {record_id}: "
                f"expected {row['tokens']}, got {tokens}"
            )

        spacy_labels = build_bio_labels(token_spans, run_spacy(nlp, text))
        presidio_labels = build_bio_labels(
            token_spans,
            run_presidio(analyzer, text, args.presidio_score_threshold),
        )

        spacy_predictions.append({"id": record_id, "predicted_labels": spacy_labels})
        presidio_predictions.append({"id": record_id, "predicted_labels": presidio_labels})

    write_jsonl(args.spacy_output, spacy_predictions)
    write_jsonl(args.presidio_output, presidio_predictions)

    summary = {
        "split_file": str(args.split_file),
        "records": len(split_rows),
        "token_mismatches": mismatches,
        "spacy_output": str(args.spacy_output),
        "presidio_output": str(args.presidio_output),
        "presidio_score_threshold": args.presidio_score_threshold,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
