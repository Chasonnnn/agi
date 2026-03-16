prepare:
	uv run prepare.py

generate-synthetic:
	uv run scripts/generate_synthetic_benchmark.py

generate-smoke-models:
	uv run scripts/generate_local_smoke_models.py

import-upchieve-math:
	uv run scripts/import_upchieve_math.py --write-provisional-action

render-redactions:
	uv run scripts/render_redactions.py --strategy typed_placeholder

smoke:
	uv run scripts/smoke_test.py

train-candidate:
	uv run train.py --stage candidate

train-action:
	uv run train.py --stage action

eval-action:
	uv run eval.py --stage action --gold data/processed/action/test.jsonl --predictions artifacts/predictions/action_test_predictions.jsonl
