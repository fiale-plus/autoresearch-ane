# ANE autoresearch log

## Current best known
- `val_loss`: **1.659077** (2026-08-21, evaluated on the full 50-shard val split)
- Lineage: the 2026-05-26 anchor (2.320954) → two 8-shard SGDR cycles → full-dataset Arc E
- Key config: Lion + `LOSS_SCALE=1024` + `EMBED_LR_SCALE=1.0` + `ACCUM_STEPS=2` + `LEARNING_RATE=3.8e-4`
- Checkpoints preserved: `ckpt_best_1.659077.bin` (best), `ckpt_E6_1.667089.bin`, `ckpt_E5_1.679940.bin`, `ckpt_E4_1.721839.bin`, `ckpt_E3_1.808984.bin`, plus prior `ckpt_best_1.800504.bin` and `ckpt_anchor_2.320954.bin`

## Validity fixes (2026-08-21, commit b42ac5f) — read before interpreting older entries
Source inspection found three issues that invalidated parts of the pre-August-21 record:
1. **Resume clobbered config**: `load_checkpoint` overwrote `lr`/`total_steps` from the checkpoint header, so every `LEARNING_RATE`/`TOTAL_STEPS` probe made while a checkpoint existed was a no-op. Fixed: config defines are now authoritative on resume.
2. **Schedule exhaustion**: `adam_t` persisted across windows and the live anchor sat at `adam_t=2611/3000` (~14% of peak LR, pinned at floor). The old "plateau" was a floored schedule, not an LR optimum. Fixed: new `--reset-schedule` flag / `ANE_RESET_SCHEDULE=1` env var performs an SGDR warm restart.
3. **Fixed sampling seed**: `srand48(42 + start_step)` replayed the identical batch order every resumed window. Fixed: wall-clock seed.

## Data expansion (2026-08-21)
`tinystories_data00.bin` first became shards 00–07 (~165M tokens; previously shard 00 only), then was expanded to all 50 shards (~1.025B tokens). The original single shard is preserved as `tinystories_data00_shard00_only.bin`; the 8-shard intermediate is `tinystories_8shard.bin`. Each expansion changes the val split, so the incumbent was calibrated before comparison.


## 8-shard data expansion details (2026-08-21)
`tinystories_data00.bin` is now the concatenation of shards 00–07 (~165M tokens; previously shard 00 only, ~20.7M tokens). The original single shard is preserved as `tinystories_data00_shard00_only.bin`. The val split moved to the last 10% of shard 07 and is ~0.09 harder: the 1.988286 checkpoint scores 2.078751 on the new split vs 1.988286 on the old one. Cross-split comparisons are not valid; use the calibrated bar.

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

## Local cycle (2026-08-21) — SGDR arc E (all 50 shards)
- Change: full TinyStories archive, shards 00–49, concatenated into `tinystories_data00.bin` (~1.025B tokens); warm restart from the 1.800504 lineage, then floor-LR continuation.
- Calibration: the 1.800504 checkpoint scored `1.835299` on the new full-dataset val split. Use this as the same-split baseline; old 1.800504 and new 1.835299 are not directly comparable.
- Results: E1 `2.042875` (mid-anneal), E2 `1.931931`, E3 `1.808984`, E4 `1.721839`, E5 `1.679940` (schedule crossed floor), E6 `1.667089`, E7 `1.659077`.
- Runtime: sustained ANE work thermally throttled this arc to 158–173 ms/step and 3.6–4.1% reported utilization, versus ~105 ms/step on the preceding 8-shard arc.
- Verdict: **keep all**; E7 improves the calibrated baseline by `0.176222`. The final floor-LR gain was `0.008012`, below the `0.01/window` continuation threshold; stop and restart for the next arc.

## Protocol going forward
1. Compare candidates only against results on the same val split (current bar for any new idea: beat `1.659077` on the full 50-shard split).
2. When per-window gains decay below ~0.01, run a fresh warm-restart arc (`ANE_RESET_SCHEDULE=1`, 3 windows) rather than more floor-LR continuations.
3. All 50 available shards are now active; further gains require better optimization, throughput, or a new dataset.
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
1. (2026-06-25 text, superseded Aug 26) Sticky baseline is now Lion + `LOSS_SCALE=1024` + `EMBED_LR_SCALE=1.0` + `ACCUM_STEPS=3` + `WEIGHT_DECAY=0.10`.
2. First confirm robustness from the same checkpoint trajectory; the fresh-restart cycle showed that checkpoint lineage is an experimental variable.
3. Config-only follow-ups: small, single-variable probes around accumulation and LR schedule. Avoid broad weight-decay neighborhood sweeps unless a stronger hypothesis appears.
4. Infra/code follow-ups from `README.md` and `updates/knowledge-sources-2026-06-25.md`: fused Metal Lion updates, embedding lookup speedup (`maderix/ANE` PR #39), dispatch-count reduction / compile-once discipline (`jmanhype/ane-lora-training`, `rustane`), and shape/tiling probes before large architecture changes.

## Aug26 cycle (branch autoresearch-ane/aug26)
- Session result: `val_loss 1.659077 -> 1.593065` in 5 keeps + 2 discards across 7 rows (keeps: R0-R2 floor continuations, R3c regen, R4; discards: R3a endpoint lost, R3b control; R3 same-anchor A/B — the ACCUM=3 arm's endpoint was lost when the anchor was restored, superseded by its regen R3c; R4 continuation).
- Key experiment: same-anchor A/B: ACCUM=3 reached `1.602377` vs `1.615093` for ACCUM=2 from the identical anchor (regen `1.603196`). Observed same-anchor metrics: 2147 vs 1963 steps, 110.5 vs 110.6 ms_per_step. The staging hypothesis (each Lion update re-stages weights, ~50 ms host->ANE; ~27% of wall at the pre-A/B 93 ms/step ACCUM=2 baseline) motivated the trial and matches its direction, but the A/B does not isolate staging as the cause.
- Reproducibility: winning arm re-run from anchor landed within 0.0008 (1.602377 vs 1.603196).

## Infra recommendations from direct private-API research (ane-api-research lab, Aug 25-26)
(Observations/hypotheses from short probes on this machine — not exhaustive. Raw probe evidence lives in that lab repo at `ane-lab/results/raw/`, not in this branch.)
These are code-level (out of config-only scope), measured on this machine (M3 Pro, macOS 26.5):
1. **Cached IOSurface bindings** (hypothesis): `_ANEClient.mapIOSurfacesWithModel:request:cacheInference:error:` persists IO bindings per program — could remove per-call binding setup overhead; it would NOT eliminate the ~50 ms changed-weight restage after each optimizer step (weights change every update). Biggest unknown for `ane_util_pct` (still ~5.7%).
2. **QoS queues**: in this workload (8-layer conv, 200 iters/stream), two streams on different queues showed no advantage over same-queue (both ~3-4x slower than one stream). Whether other workloads see queue-level parallelism is untested. Do not spend effort on concurrent-stream designs; do use QoS to prioritize interactive evals.
3. **RealTime lane**: `beginRealTimeTask` returned NO in our unentitled CLI process; cause unknown (no realtime-specific entitlement constant exists in `_ANEStrings`).
4. **VisionCoreE5RT compile options** expose `fullyANEResident`, raw `customCompilationOptions`, multi-entry MIL programs (`milEntryPoints`) — a route to fewer dispatches than the current 10-kernel pipeline if we ever recompile the trainer's graph.
5. **Perf counters** (`_ANEPerformanceStats`) were not populated even with `perfStatsMask=0xFFFFFFFF`; cause unknown (aned-side enablement is one hypothesis). Would give a true hw-time vs wall-time split if enabled.
6. **SRAM working set**: the kernel-bench tooling warns above 32 MB working set; treat that warning as the observed bound and keep DIM*SEQ activation footprint under it when probing architecture changes (spill behavior itself was not directly measured).

### Correction to the staging story above (exact-code measurement, Aug 26 lab)
Direct benchmarking with the trainer's OWN implementations verbatim
(`transpose_weight` = vDSP_mtrans, `lion_update` from stories_cpu_ops.h, exact
shapes) measures ~49.4 ms per update: transposes 8.3 ms + Lion matrices 20.9 ms +
Lion full-32k embedding 9.3 ms + f32->f16 staging conversion ~11.0 ms — matching
the repo's "~50 ms" figure. The dominant single item is the scalar sign loop in
lion_update plus updating all 32k embedding rows under USE_VOCAB_COMPACT (~23k
rows get no gradient). Evidence-backed levers (code-level): restrict optimizer to
active embedding rows (~6.6 ms), chunk lion_update's scalar tail across threads,
prune the four dead kernels (sdpaFwd/woFwd/qBwd/kvBwd are staged but never
evaluated; ~2.7 ms + compile/memory). v2 harness (verbatim structure, real IOSurfaces, checksum-validated) measures update block 39.05 ms + staging 15.95 ms current / 11.98 ms pruned => ~55 ms total; pruning the four dead kernels saves ~4.0 ms/update (+~1.2% steps).
The ACCUM=3 A/B result stands empirically. Full measurements:
ane-api-research lab, `ane-lab/results/staging-anatomy.md` + `results/raw/exact-bench.txt`.

### Wave 2 (Aug 26, continued): ACCUM_STEPS=4 adopted
- Same-anchor A/B #2: ACCUM=4 reached `1.571062` vs `1.581343` for ACCUM=3
  (regen `1.577762`; arms ran 2245 vs 2139 steps at ~110.7 ms_per_step).
  ACCUM=4 adopted; sticky config now `ACCUM_STEPS=4`.
- Trend: each +1 ACCUM keeps winning on this full-data lineage (2 -> 3 -> 4),
  consistent with fewer weight-restagings per token and larger effective batch.
  Next bounded probe: ACCUM=5 same-anchor A/B; stop when an A/B loses.
- Protocol fix: arm endpoints are now cloned (`ckpt.arm_*.bin`) BEFORE restoring
  the anchor — R3a and R6a endpoints were lost by restoring first.
