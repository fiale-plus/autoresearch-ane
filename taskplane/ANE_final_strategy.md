# ANE Research Strategy & Execution Guide (For Hardware with NVIDIA GPU)

**Project Goal:** Achieve the lowest possible `val_loss` on the TinyStories dataset by autonomously tuning hyperparameters in `ane/experiment_config.h` and optimizing the training pipeline.

**Metric:** `val_loss` (Cross-Entropy). Lower is better.

**Target Hardware:** NVIDIA GPU (H100 recommended, but any GPU supporting CUDA/PyTorch 2.9.1+cu128 is required).

---

## ⚙️ Phase 0: Environment Setup (MUST BE DONE FIRST)

**Prerequisites:**
1.  Python 3.10+ environment with `uv` installed.
2.  PyTorch version compatible with CUDA 12.8 (e.g., `torch==2.9.1+cu128`).
3.  All dependencies installed: `uv sync`.
4.  Data downloaded: `bash ane/download_data.sh`.
5.  Binary compiled: `make -C ane train_ane`.

**Baseline Measurement:**
Execute the initial run to establish the starting point:
`python harness_ane.py`
**Record the resulting `val_loss` and the exact command used.** This is the "Cycle 0" baseline.

---

## 🔬 Phase 1: Autonomous Research Cycle (The Loop)

Use the `/autoresearch` skill targeting the `ane/` directory. The agent must strictly adhere to the following cycle discipline:

**Cycle Structure:**
1.  **MEASURE:** Run `python harness_ane.py`. Parse the output to get the current `val_loss`.
2.  **IDENTIFY:** Analyze `ane/autoresearch.md` and `ane/autoresearch.jsonl`. Determine the single most promising, un-plateaued area for improvement.
3.  **IMPLEMENT:** Modify **only** `ane/experiment_config.h` with a small, targeted change (e.g., adjust `LEARNING_RATE`, change `SOFTCAP`, or modify an architecture parameter like `HEADS`).
4.  **BUILD:** Run `ane/autoresearch.checks.sh` (if available/necessary) to ensure the change compiles and passes basic tests. **If this fails, revert immediately and log `status: "checks_failed"`**.
5.  **TEST:** Re-run `python harness_ane.py`. Compare the new `val_loss` to the previous cycle's best.
6.  **LOG:** Append a JSONL entry to `ane/autoresearch.jsonl` detailing the change, before/after metrics, delta, and verdict (`keep`/`revert`/`needs_more_data`).
7.  **COMMIT:** If `action: "keep"`, commit the change to Git with a descriptive message: `autoresearch: [description] (val_loss -[delta])`. If `action: "revert"`, use `git clean -fd` after reverting the file.
8.  **UPDATE STRATEGY:** Rewrite `ane/autoresearch.md` with the new state, what worked, and the next 1-3 hypotheses.

**Critical Constraints to Enforce:**
*   **Hardware Constraints:** Never propose an architecture that violates the 512-channel limit or the 16-weight limit.
*   **Hyperparameter Tuning:** Prioritize tuning the following knobs based on recent findings:
    *   `LEARNING_RATE` (Base LR)
    *   `WEIGHT_DECAY`
    *   `SOFTCAP`
    *   `EMBED_LR_SCALE` (Aim for 1.0)
*   **Optimization:** Focus on the **Lion optimizer** as it proved superior to Adam in this specific ANE context.

---

## ✅ Phase 2: Finalization & PR Creation

Once the cycle plateaus or the target loss is achieved:
1.  **Final Review:** Manually review `ane/autoresearch.jsonl` to ensure all successful steps are logged.
2.  **Documentation:** Update `ane/autoresearch.md` with the final conclusion, the best hyperparameters found, and a summary of the key insights (e.g., "The combination of Lion + LOSS_SCALE=1024 + EMBED_LR_SCALE=1.0 achieved the best result").
3.  **Commit:** Commit all final artifacts (`.jsonl`, `.md`, `experiment_config.h`).
4.  **PR:** Trigger the final integration/PR, explaining the entire journey: *What was the initial state, what was the hypothesis, what was the mechanism of improvement (e.g., "Fusing kernels reduced I/O overhead"), and what is the final achieved metric.*