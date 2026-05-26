# ANE autoresearch log

## Current best known
- `val_loss`: **2.320954**
- Key config: Lion + `LOSS_SCALE=1024` + `EMBED_LR_SCALE=1.0` + `ACCUM_STEPS=2`

## Latest local cycle
- Change: `ACCUM_STEPS 4 -> 2` (from the fresh anchor checkpoint; best-known LR/WD/LR_MIN unchanged)
- Context: resumed from the fresh anchor checkpoint to test lower effective batch / update frequency
- Result: `val_loss 2.320954`, `train_loss 1.880684`, `steps 2270`, `ms_per_step 93.7`, `ane_util_pct 6.7`
- Verdict: **keep**; new best and materially better than the prior 2.432 baseline

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

## Next hypothesis
1. Revert `WEIGHT_DECAY` to `0.10` and stop sweeping this neighborhood; it appears unstable.
2. If another cycle is needed, return to the best-known config and re-run a clean confirmatory cycle.
