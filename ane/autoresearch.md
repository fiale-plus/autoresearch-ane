# ANE autoresearch log

## Current best known
- `val_loss`: **1.800504** (2026-08-21, evaluated on the 8-shard val split)
- Lineage: the 2026-05-26 anchor (2.320954) → two full SGDR warm-restart cycles on shards 00–07
- Key config: Lion + `LOSS_SCALE=1024` + `EMBED_LR_SCALE=1.0` + `ACCUM_STEPS=2` + `LEARNING_RATE=3.8e-4`
- Checkpoints preserved: `ckpt_best_1.800504.bin` (best), `ckpt_best_1.836396.bin`, `ckpt_best_1.869944.bin`, `ckpt_best_1.988286.bin`, `ckpt_anchor_2.320954.bin` (pre-session anchor)

## Validity fixes (2026-08-21, commit b42ac5f) — read before interpreting older entries
Source inspection found three issues that invalidated parts of the pre-August-21 record:
1. **Resume clobbered config**: `load_checkpoint` overwrote `lr`/`total_steps` from the checkpoint header, so every `LEARNING_RATE`/`TOTAL_STEPS` probe made while a checkpoint existed was a no-op. Fixed: config defines are now authoritative on resume.
2. **Schedule exhaustion**: `adam_t` persisted across windows and the live anchor sat at `adam_t=2611/3000` (~14% of peak LR, pinned at floor). The old "plateau" was a floored schedule, not an LR optimum. Fixed: new `--reset-schedule` flag / `ANE_RESET_SCHEDULE=1` env var performs an SGDR warm restart.
3. **Fixed sampling seed**: `srand48(42 + start_step)` replayed the identical batch order every resumed window. Fixed: wall-clock seed.

## Data expansion (2026-08-21)
`tinystories_data00.bin` is now the concatenation of shards 00–07 (~158M tokens; previously shard 00 only, ~19.7M tokens). The original single shard is preserved as `tinystories_data00_shard00_only.bin`. The val split moved to the last 10% of shard 07 and is ~0.09 harder: the 1.988286 checkpoint scores 2.078751 on the new split vs 1.988286 on the old one. Cross-split comparisons are not valid; use the calibrated bar.

## Local cycle (2026-08-21) — SGDR arc A (shard-00 data)
- Change: warm restart (`adam_t 2611 → 0`) from the 2.320954 anchor, then two continuation windows.
- Results: A1 `2.457641` (mid-anneal), A2 `2.113512`, A3 `1.988286` (anneal complete).
- Verdict: **keep all**; A3 beat the old best by −0.333 and confirmed the schedule-exhaustion diagnosis.

## Local cycle (2026-08-21) — SGDR arc B (+ 8-shard data)
- Change: warm restart from the 1.988286 lineage on the expanded dataset, then continuations.
- Results: B1 `2.258588` (mid-anneal), B2 `1.996975`, B3 `1.869944` (anneal complete), B4 `1.836396` (floor-LR continuation).
- Verdict: **keep all**; B4 shows floor-LR training still improves when unseen tokens remain — the opposite of the August small-data probes, where it degraded.

## Local cycle (2026-08-21) — SGDR arc C (second restart)
- Change: warm restart from the 1.836396 lineage, then two continuations.
- Results: C1 `2.143164` (mid-anneal), C2 `1.919288`, C3 `1.800504` (anneal complete).
- Verdict: **keep all**; C3 is the all-time best. Restart arcs currently yield ~−0.03..−0.05 per full cycle; continuation windows after an arc yield <−0.01.

## Protocol going forward
1. Compare candidates only against results on the same val split (current bar for any new idea: beat `1.800504`).
2. When per-window gains decay below ~0.01, run a fresh warm-restart arc (`ANE_RESET_SCHEDULE=1`, 3 windows) rather than more floor-LR continuations.
3. There are still 42 unused shards (`data08`–`data49` in the HF archive); expanding data further is the cheapest known lever.
4. `ms_per_step` excludes the optimizer/restage block, which consumes ~27% of wall time — see `updates/analysis-2026-08-21.md` for the remaining Tier-2 levers.

## Latest local cycle (2026-08-20)
- Change: bounded probes from a preserved copy of the 2.320954 checkpoint: `ACCUM_STEPS 2 -> 1`, then `LEARNING_RATE 3.8e-4f -> 3.6e-4f`, then `3.7e-4f`.
- Context: each probe used `ANE_WALL_TIME=300`; the anchor checkpoint was restored before every probe so candidate trajectories were comparable.
- Results: `2.401719` (`ACCUM=1`, 1123 steps, 137.7 ms/step), `2.346754` (`LR=3.6e-4`, 2137 steps, 98.4 ms/step), and `2.348575` (`LR=3.7e-4`, 2174 steps, 97.0 ms/step).
- Verdict: **discard all three**; the retained configuration remains unchanged.


## Prior local cycle
- Change: `ACCUM_STEPS 7 -> 4` (from the fresh-restart checkpoint; best-known LR/WD/LR_MIN unchanged)
- Context: resumed from the fresh anchor checkpoint to test lower effective batch / update frequency
- Result: `val_loss 2.455971`, `train_loss 1.970962`, `steps 2621`, `ms_per_step 93.6`, `ane_util_pct 6.7`
- Verdict: keep; improved again and is now close to the best

## Prior local cycle
- Change: `ACCUM_STEPS 14 -> 7` (from the fresh-restart checkpoint; best-known LR/WD/LR_MIN unchanged)
- Context: resumed from the fresh anchor checkpoint to test optimization trajectory sensitivity
- Result: `val_loss 2.664419`, `train_loss 2.262500`, `steps 2807`, `ms_per_step 93.7`, `ane_util_pct 6.7`
- Verdict: keep; better than the fresh-restart continuation but still above the 2.432 best

## Prior local cycle
- Change: continued training from the fresh restart with no config change
- Context: resume from the new checkpoint created by the fresh best-config anchor run
- Result: `val_loss 3.147180`, `train_loss 2.946599`, `steps 2944`, `ms_per_step 93.6`, `ane_util_pct 6.7`
- Verdict: keep as a partial recovery, but still far from the 2.432 best

## Prior local cycle
- Change: **fresh restart** at the best-known config (`LEARNING_RATE 3.8e-4`, `WEIGHT_DECAY 0.10`, `LR_MIN_FRAC 0.10`, `ACCUM_STEPS=14`)
- Context: deleted checkpoint and re-ran from scratch to test reproducibility/anchor the baseline
- Result: `val_loss 3.651091`, `train_loss 3.271285`, `steps 2945`, `ms_per_step 93.7`, `ane_util_pct 6.7`
- Verdict: **discard**; fresh restart did not reproduce the near-best and suggests checkpoint trajectory matters

## Prior local cycle
- Change: `LR_MIN_FRAC 0.1 -> 0.05` (with `LEARNING_RATE 3.8e-4`)
- Context: resumed from existing checkpoint after `ACCUM_STEPS=14`
- Result: `val_loss 2.675144`, `train_loss 1.440777`, `steps 2943`, `ms_per_step 93.7`, `ane_util_pct 6.7`
- Verdict: discard; materially worse than the recent 3.8e-4 run

## Earlier local cycle
- Change: `LEARNING_RATE 3.8e-4 -> 3.7e-4`
- Context: resumed from existing checkpoint after `ACCUM_STEPS=14`
- Result: `val_loss 2.514317`, `train_loss 2.242823`, `steps 2946`, `ms_per_step 93.6`, `ane_util_pct 6.7`
- Verdict: discard; worse than the recent 3.8e-4 run and still above the all-time best

## Earlier local cycle
- Change: `LEARNING_RATE 4.0e-4 -> 3.8e-4`
- Context: resumed from existing checkpoint after `ACCUM_STEPS=14`
- Result: `val_loss 2.437545`, `train_loss 1.781786`, `steps 2947`, `ms_per_step 93.6`, `ane_util_pct 6.7`
- Verdict: **keep**; essentially matched the best and only missed by 0.0055

## Earlier local cycle
- Change: `LEARNING_RATE 5e-4 -> 4.0e-4`
- Context: resumed from existing checkpoint after `ACCUM_STEPS=14`
- Result: `val_loss 2.461927`, `train_loss 2.252364`, `steps 2941`, `ms_per_step 93.7`, `ane_util_pct 6.7`
- Verdict: keep; close to best but still not top

## Earlier local cycle
- Change: `LEARNING_RATE 5e-4 -> 4.25e-4`
- Context: resumed from existing checkpoint after `ACCUM_STEPS=14`
- Result: `val_loss 2.644823`, `train_loss 2.964848`, `steps 2916`, `ms_per_step 94.6`, `ane_util_pct 6.6`
- Verdict: keep for now, but not an all-time best

## Source-informed next hypotheses (updated 2026-06-25)
1. Keep the current best-known config as the sticky baseline: Lion + `LOSS_SCALE=1024` + `EMBED_LR_SCALE=1.0` + `ACCUM_STEPS=2` + `WEIGHT_DECAY=0.10`.
2. First confirm robustness from the same checkpoint trajectory; the fresh-restart cycle showed that checkpoint lineage is an experimental variable.
3. Config-only follow-ups: small, single-variable probes around accumulation and LR schedule. Avoid broad weight-decay neighborhood sweeps unless a stronger hypothesis appears.
4. Infra/code follow-ups from `README.md` and `updates/knowledge-sources-2026-06-25.md`: fused Metal Lion updates, embedding lookup speedup (`maderix/ANE` PR #39), dispatch-count reduction / compile-once discipline (`jmanhype/ane-lora-training`, `rustane`), and shape/tiling probes before large architecture changes.
