# Knowledge Sources Update — 2026-06-25

## Repository sync state

- Fetched `origin` and `upstream` with pruning.
- Rebased local `master` onto `origin/master`; remote-only merge commit `e48ec12` is now the base.
- Local `master` remains ahead because it preserves the local autoresearch log commit (`ane/autoresearch.md` / `ane/autoresearch.jsonl`) plus this documentation update.
- Latest ANE autoresearch result in this checkout: **val_loss 2.320954** with Lion, `LOSS_SCALE=1024`, `EMBED_LR_SCALE=1.0`, `ACCUM_STEPS=2`.

## Current ANE/autoresearch baseline

- **Best local result:** `val_loss 2.320954`, `train_loss 1.880684`, `steps 2270`, `ms_per_step 93.7`, `ane_util_pct 6.7`.
- **Best-known config:** Lion optimizer, `LOSS_SCALE=1024`, `EMBED_LR_SCALE=1.0`, `ACCUM_STEPS=2`; keep `WEIGHT_DECAY=0.10` and avoid further nearby WD sweeps unless a stronger hypothesis appears.
- **Autoresearch lesson:** checkpoint trajectory matters. A fresh restart at the prior best config did not reproduce the near-best run, while lower accumulation from the anchored checkpoint improved to the new best.

## New ecosystem findings since 2026-04-09

### 1. `ncdrone/rustane` — Rust-native hybrid ANE + Metal training/inference

**What changed:** A newer Rust engine reports end-to-end ANE training validation from 579M to 5B parameters, plus forward-only probes up to 30B on M4 Max 128GB.

Key claims from the README:
- Training pipeline validates compile + forward + backward + Adam + loss decrease.
- Best validated training scale: **5.0B params**, ~6,893 ms/step, ~74 tok/s, ~85GB RAM.
- Forward-only probes: 5B ~2.1s, 7B ~3.1s, 10B ~4.7s, 30B ~75s; reported limit is RAM rather than an ANE compilation ceiling.
- Architecture finding: wide+shallow wins below ~3B; deep+narrow wins above; **dim=5120 is an efficiency cliff** and dim <=4096 is recommended.

**Actionability:** ⭐⭐⭐ HIGH as a design reference, ⭐ LOW as a drop-in dependency.
- Confirms ANE training can scale far beyond the 110M TinyStories setup if memory/orchestration are engineered carefully.
- For this repo’s 5-minute TinyStories loop, the actionable bit is not scale-up immediately; it is architecture shape discipline: avoid dimensions that trigger tiling/cliffs, and prefer dispatch-count-aware depth/width sweeps.
- Rust/Metal implementation is a separate stack, so porting is not config-only.

### 2. `jmanhype/ane-lora-training` — updated hybrid MLX + ANE LoRA results

**What changed:** Repo is active through June 2026 and now reports a persistent bridge plus measured ANE power correction.

Key claims from the README:
- Qwen2.5-3B-Instruct-4bit LoRA, rank 8, last 2 layers, 446K trainable params.
- Loss improves **4.07 → 2.64** over 200 steps.
- Fused 4-matmul LoRA gradient kernel: **0.36 ms/module, 1 dispatch** vs 0.72 ms/module, 4 dispatches unfused.
- Persistent bridge step time ~155 ms; compile-once kernel ~100 ms.
- ANE power during training measured at ~37–47 mW; GPU remains the high-power component for MLX inference.
- Spatial dimension constraint: last dimension must be >=16 and multiple of 16.

**Actionability:** ⭐⭐ MEDIUM.
- Reinforces the highest-leverage infra theme: fuse dispatches and compile once.
- The LoRA split (MLX forward, ANE gradients) is not directly this repo’s full-training path, but the packed-IOSurface dynamic matmul and dispatch fusion patterns are relevant.
- The spatial multiple-of-16 constraint should stay in the ANE constraints table next to our multiple-of-32 matmul-inner-dim rule.

### 3. `harsha-gouru/apple-neural-engine-notes` — planning heuristics for ANE architecture fit

**What changed:** New public research notes summarize ANE training/inference experiments and hybrid runtime conclusions.

Key claims from the README:
- Software overhead, not raw ANE compute, is often the real bottleneck: layout conversion, dispatch boundaries, cache handling, compile/setup cost.
- Architecture fit matters more than parameter count: recurrent/state-space and hybrid recurrent+attention models are the best fit; pure transformers are acceptable only when small/bounded.
- SRAM heuristic: keep the largest layer matrix inside the effective on-chip budget when possible; otherwise assume tiling/layout penalties.
- Practical model map: 0.5B–2B strong FP16 fit; 3B–4B promising with quantization; 7B–12B research-only; 27B+ poor target for this path.

**Actionability:** ⭐⭐ MEDIUM.
- Supports future experiments beyond pure GPT blocks: bounded attention, recurrent/state-space blocks, or hybrid layers may be better ANE targets.
- For this repo, keep near-term sweeps conservative; major architecture changes should include explicit shape/tiling probes.

### 4. `harsha-gouru/ane-gmlp-research` — on-device ANE training constraints from gMLP

**What changed:** New public write-up reports end-to-end ANE training of a small gMLP and documents trainer constraints.

Key claims from the README:
- Trained 153K-parameter gMLP to **87.3% train / 81.2% test accuracy** over 200 epochs in ~16 seconds on M4 Pro ANE.
- gMLP was selected because it avoids attention/softmax gradient issues and maps to matmul-heavy ANE ops.
- Important architectural fix: leading identity projection to normalize internal tensor layout before residual paths.
- Freezing base blocks and training LoRA-style adapters avoids weight-desync issues seen with broader ANE weight updates.

**Actionability:** ⭐ LOW/MEDIUM.
- Small-classification setup is not directly TinyStories LM training.
- Still useful as evidence that tensor layout and residual path compatibility can dominate apparent shape compatibility; future ANE architecture changes should validate gradients op-by-op.

### 5. `slavko-at-klincov-it/ANE-Training` — still core reference, no newer main commits after Apr 5

**Status update:** Main branch remains at Apr 5, 2026, but it is still one of the strongest references for this repo.

Key claims kept from prior scan:
- Standalone `libane` C API, dynamic spatial packing, compile-once execution.
- Stories-110M training result: loss 1.86 and coherent sample text; Tiny-ANE-15M loss 2.35.
- Measured time split: ANE ~24%, CPU/AMX ~59%, IOSurface transfer ~9%, other overhead ~8%.
- Metal fused Adam pattern reduced optimizer overhead in their pipeline; for us, a Metal fused Lion kernel remains a Tier-2 infra idea.

**Actionability:** ⭐⭐⭐ HIGH.
- Keep `LOSS_SCALE=1024` as the default/sticky finding; it already contributed to this repo’s best local run.
- Fused optimizer/gradient infrastructure is the likely next non-config breakthrough after accumulation/LR sweeps plateau.

### 6. `maderix/ANE` issues/PRs — constraint updates

**Status update:** Main branch remains Mar 10, 2026. Relevant items:
- PR #39 embedding speedup remains open and still last updated Mar 4.
- PR #40 E5 runtime/custom MIL research remains open and still last updated Mar 4.
- Issue #42 M3 Ultra 512-channel hard constraint is still open and was updated Jun 24.
- PR #45 security fix remains open.

**Actionability:** ⭐⭐ MEDIUM.
- Embedding speedup remains a concrete Tier-2 port candidate.
- Constraint table should keep 512-channel/M3 Ultra and mixed-weight/kernel-layout notes explicit.

### 7. Upstream `karpathy/autoresearch`

**Status update:** `master` still points at `228791f` from Mar 26, 2026. No new upstream training techniques or architecture changes were found in the default branch.

## Updated next-action plan

### Tier 1 — config-only / current rules

1. Preserve the best-known config unless testing a single focused hypothesis: Lion + `LOSS_SCALE=1024` + `EMBED_LR_SCALE=1.0` + `ACCUM_STEPS=2` + `WEIGHT_DECAY=0.10`.
2. Run a confirmatory cycle from the same checkpoint trajectory before declaring the 2.320954 result robust.
3. If sweeping, prefer small changes to accumulation or LR schedule; avoid broad WD neighborhood sweeps, which were unstable.

### Tier 2 — infra/code changes

1. Metal fused Lion kernel, borrowing the fused optimizer dispatch pattern from `ANE-Training`.
2. Embedding lookup speedup from `maderix/ANE` PR #39.
3. Dispatch-count reduction and compile-once discipline from `jmanhype/ane-lora-training` and `rustane`.
4. Shape/tiling probe harness before larger width/depth changes; avoid the `rustane` dim=5120 cliff class of issue.

### Tier 3 — research directions

1. ANE-friendly alternatives to pure GPT blocks: gMLP, bounded attention, or recurrent/state-space hybrids.
2. Adapter/frozen-base experiments if full-weight updates continue showing checkpoint trajectory fragility.
3. Quantized/int8-aware larger model probes only after dynamic pipeline compatibility is proven.

## Source repositories scanned

| Repo | Last observed commit/update | Relevance | Key takeaway |
|------|-----------------------------|-----------|--------------|
| [`ncdrone/rustane`](https://github.com/ncdrone/rustane) | updated 2026-06-25 | ⭐⭐⭐ | Rust ANE+Metal engine; training validation to 5B, forward probes to 30B, dim/architecture cliffs documented |
| [`jmanhype/ane-lora-training`](https://github.com/jmanhype/ane-lora-training) | updated 2026-06-19 | ⭐⭐ | Fused LoRA gradient dispatch, persistent bridge, packed IOSurface dynamic matmul, spatial multiple-of-16 constraint |
| [`harsha-gouru/apple-neural-engine-notes`](https://github.com/harsha-gouru/apple-neural-engine-notes) | updated 2026-05-21 | ⭐⭐ | Architecture-fit and SRAM heuristics; software overhead dominates many ANE workloads |
| [`harsha-gouru/ane-gmlp-research`](https://github.com/harsha-gouru/ane-gmlp-research) | updated 2026-05-21 | ⭐ | gMLP on-device training notes; tensor layout/residual compatibility and adapter training lessons |
| [`slavko-at-klincov-it/ANE-Training`](https://github.com/slavko-at-klincov-it/ANE-Training) | main 2026-04-05; repo updated 2026-05-16 | ⭐⭐⭐ | `libane`, Stories-110M results, fused optimizer pattern, time-split benchmarks |
| [`maderix/ANE`](https://github.com/maderix/ANE) | main 2026-03-10; issue #42 updated 2026-06-24 | ⭐⭐ | Embedding PR #39, E5 runtime PR #40, 512-channel M3 Ultra constraint |
| [`karpathy/autoresearch`](https://github.com/karpathy/autoresearch) | master 2026-03-26 | ⭐ | No newer upstream training technique on default branch |
| [`fiale-plus/ANE`](https://github.com/fiale-plus/ANE) | main 2026-03-10 | ⭐⭐ | Local fork mirrors `maderix/ANE` state; INT8 notes still not drop-in viable for this dynamic pipeline |
