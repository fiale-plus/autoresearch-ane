# ANE Research Task Plane

This file structures the execution of the ANE research cycle.

## Task 1: Setup and Baseline Measurement
**Goal:** Establish the initial `val_loss` baseline for the ANE training stack.
**Steps:**
1.  `uv sync` (Ensure dependencies are correct).
2.  `bash ane/download_data.sh` (Download TinyStories data).
3.  `make -C ane train_ane` (Compile the training binary).
4.  `python harness_ane.py` (Run initial experiment to get baseline `val_loss`).
**Output:** Baseline `val_loss` and confirmation that the environment is ready for iterative tuning.

## Task 2: Autonomous Optimization Cycle
**Goal:** Iteratively reduce `val_loss` by tuning `ane/experiment_config.h`.
**Tool:** /autoresearch
**Target:** `ane/`
**Metric:** `val_loss`
**Cycles:** Run until plateau or target reached.

## Task 3: Finalization
**Goal:** Create PR summarizing findings.
**Steps:**
1.  Review `ane/autoresearch.jsonl` for all changes.
2.  Update `ane/autoresearch.md` with final strategy and results.
3.  Commit all changes and trigger the final PR.