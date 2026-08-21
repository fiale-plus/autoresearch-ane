# ANE Research Setup Task

This task sets up the environment and establishes the baseline for autonomous research on the Apple Neural Engine (ANE) training stack, as detailed in the README.md.

**Goal:** Achieve the lowest possible `val_loss` on the TinyStories dataset by autonomously tuning hyperparameters in `ane/experiment_config.h` and optimizing the training pipeline.

**Metric:** `val_loss` (Cross-Entropy). Lower is better.

**Scope:** The entire ANE training stack within the `ane/` directory.

**Prerequisites:**
1.  The system must have access to an Apple Silicon GPU (M-series chip).
2.  Dependencies must be installed via `uv sync`.
3.  Data must be downloaded via `bash ane/download_data.sh`.
4.  The training binary must be compiled via `make -C ane train_ane`.

**Initial Task Steps (Baseline Measurement):**
1.  Execute the setup commands sequentially to ensure the environment is ready for the first cycle.
2.  Run the baseline experiment using the default configuration in `ane/experiment_config.h` to establish the initial `val_loss` baseline.

**Expected Output:**
- A successful execution of the setup commands.
- A measurable `val_loss` value from the initial run, which will serve as the "before" metric for the first autoresearch cycle.

**Constraint:** All subsequent changes must be confined to modifying `ane/experiment_config.h` and must be validated by running the full cycle (Measure -> Identify -> Implement -> Build -> Test -> Log).

**Next Step:** Once the baseline is established, I will initiate the `/autoresearch` skill, pointing it to the `ane/` directory, to begin the iterative optimization process.