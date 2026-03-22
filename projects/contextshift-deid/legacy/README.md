# Legacy Archive

This folder keeps superseded datasets, checkpoints, and experiment artifacts that are still useful for record-keeping but should not be treated as the current default workspace state.

Current root locations should hold:

- active benchmark datasets under `data/processed/`
- active experiment families under `artifacts/experiments/`
- current checkpoints under `runs/`

Archived under `legacy/`:

- proxy candidate datasets from the UpChieve adaptation sprint
- superseded March 14-15 experiment artifacts
- the old placeholder `runs/candidate` checkpoint
- supporting annotation pools for the archived proxy-candidate workflow

If you need to rerun one of the archived workflows, use the updated scripts and docs that now point to these `legacy/` paths explicitly.
