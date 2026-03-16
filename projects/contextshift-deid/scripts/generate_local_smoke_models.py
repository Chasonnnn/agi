from __future__ import annotations

from pathlib import Path

from transformers import BertConfig, BertForSequenceClassification, BertForTokenClassification, BertTokenizerFast

ROOT = Path(__file__).resolve().parents[1]
SMOKE_MODELS_DIR = ROOT / "artifacts" / "smoke_models"

VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    ".",
    ",",
    ":",
    "-",
    "_",
    "@",
    "/",
    "math",
    "history",
    "literature",
    "student",
    "teacher",
    "name",
    "email",
    "phone",
    "school",
    "macbeth",
    "shakespeare",
    "mr",
    "shah",
    "lincoln",
    "high",
    "what",
    "my",
    "is",
    "the",
    "context",
    "span",
    "subject",
    "anchor",
    "speaker",
    "role",
    "entity",
    "type",
    "semantic",
    "intent",
]


def _write_vocab(path: Path) -> None:
    path.write_text("\n".join(VOCAB) + "\n", encoding="utf-8")


def _build_tokenizer(model_dir: Path) -> BertTokenizerFast:
    vocab_path = model_dir / "vocab.txt"
    _write_vocab(vocab_path)
    tokenizer = BertTokenizerFast(vocab_file=str(vocab_path), do_lower_case=True)
    tokenizer.save_pretrained(str(model_dir))
    return tokenizer


def _build_candidate_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _build_tokenizer(model_dir)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=256,
        num_labels=3,
    )
    model = BertForTokenClassification(config)
    model.save_pretrained(str(model_dir))


def _build_action_model(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _build_tokenizer(model_dir)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=512,
        num_labels=3,
    )
    model = BertForSequenceClassification(config)
    model.save_pretrained(str(model_dir))


def main() -> None:
    candidate_dir = SMOKE_MODELS_DIR / "candidate-bert"
    action_dir = SMOKE_MODELS_DIR / "action-bert"
    _build_candidate_model(candidate_dir)
    _build_action_model(action_dir)
    print(f"Local smoke models written to {SMOKE_MODELS_DIR}")


if __name__ == "__main__":
    main()
