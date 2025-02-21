# TinyCMax

Minimal implementation of the contrast maximization (CMax) framework for self-supervised learning (SSL) from events. Related publications:

- ...

## Installation

```
conda env create -f env.yaml && conda activate tinycmax && pre-commit install
```

## Usage

Training:
```
python train.py
```
- Logging to Weights & Biases:
    - Create `logs` folder in repo root
    - Run command with `logger=wandb logger.notes="some notes"` added
- Visualize in Rerun:
    - Run `rerun` in separate window (with environment activated)
    - Set `server=127.0.0.1:9876` in [`live_vis.yaml`](config/callbacks/live_vis.yaml)
    - Run command with `+callbacks=live_vis` added

Validation:
```
python validate.py runid=<wandb_run_id>
```
- Run selection:
    - Get `wandb_run_id` from Weights & Biases
    - Add `checkpoint=<checkpoint_id>` to select specific checkpoint
- Visualize in Rerun: same as above
