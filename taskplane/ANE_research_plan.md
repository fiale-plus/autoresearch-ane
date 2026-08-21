# ANE Research Task Plan

This plan outlines the execution of the ANE research cycle using the autoresearch framework.

## 1. Setup & Baseline (Manual/Pre-task)
*   **Action:** Run setup commands (download data, compile binary).
*   **Output:** Initial `val_loss` baseline.

## 2. Core Loop (Autoresearch)
*   **Skill:** /autoresearch
*   **Target:** `ane/` directory
*   **Metric:** `val_loss`
*   **Cycle Goal:** Systematically improve `val_loss` by modifying `ane/experiment_config.h`.
*   **Key Focus Areas (from README):**
    *   Hyperparameter tuning (e.g., `LEARNING_RATE`, `WEIGHT_DECAY`, `SOFTCAP`).
    *   Architecture exploration (e.g., `DIM`, `HEADS`, `NLAYERS`) while respecting hardware constraints (512 channels, 16 weight limit).
    *   Optimizer selection (Lion vs Adam).
*   **Constraint Adherence:** Must follow the full cycle discipline (Measure -> Identify -> Implement -> Build -> Test -> Log).

## 3. Finalization
*   Once the target `val_loss` is reached or plateaued, the process will conclude by generating a final report and creating a PR.