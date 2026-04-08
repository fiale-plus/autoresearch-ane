# Knowledge Sources Update — 2026-04-09

## Current State (main branch, `origin/master`)

**Best result:** val_loss 2.489 at 96.7 ms/step (M4 Max, 6L/768d/12h, Lion, ACCUM=12, Loss Scale=512, Softcap=30)
**Branch:** `autoresearch-ane/mar28` — best experiment run
**Config:** DIM=768, HIDDEN=2048, HEADS=12, NLAYERS=6, SEQ=512, Lion optimizer, vocab compact, GQA code present but N_KV_HEADS=HEADS (must keep — GQA with fewer KV heads crashes MIL compiler)

## New Ecosystem Findings Since 2026-03-28

### 1. Metal Fused Adam — `slavko-at-klincov-it/ANE-Training`

**What:** GPU-accelerated fused Adam optimizer using Metal compute shaders.
- **33.7ms for 110M params** vs CPU ~77ms (2.3x faster)
- Float4-vectorized kernel: fused unscaling + moment update + param update + grad zeroing in single dispatch
- Zero-copy via `MTLBuffer` (SharedMode)
- `adam_kernel.metal` + `metal_adam.h` — ~300 lines total
- **Status:** Proven, integrated into their training pipeline. M4 Mini benchmarks: 84ms/step with Metal Adam (was 93.1ms CPU)

**Actionability:** ⭐⭐⭐ HIGH
- Our Lion optimizer runs on CPU (already vectorized with vDSP), but still a significant fraction of the 96ms/step budget
- Our Adam code does 3x malloc/free per weight matrix (g_sq, mhat, vhat) — Metal kernel eliminates all mallocs
- **HOWEVER:** We use Lion (sign-based, no v buffer), which is simpler than Adam. The Metal speedup for Lion would be smaller since there's no v accumulation.
- **Adoptable path:** Port `metal_adam.h` + `adam_kernel.metal` into `ane/`. Add Metal Lion kernel. Keep as compile-time option.
- **Constraint check:** `program_ane.md` says "Modify `ane/experiment_config.h` — this is the ONLY file you edit." Would need to update program to allow Metal infra changes to `train_ane.m` + `stories_cpu_ops.h`.
- **Estimated impact:** ~5-10ms/step savings (CPU Lion is ~20ms, Metal could be ~10ms)

### 2. Slavko ANE-Training — Comprehensive Benchmark & Training Infra

**What:** Full `libane` C API, hardware monitoring, model sweep, 5000-step training runs.
- **M4 Mini benchmarks (Stories-110M):** 93ms/step (sequential), 71.8ms/step (pipeline), 13.86 peak TFLOPS
- **Training convergence:** Tiny-ANE-15M to loss 3.25 (10K steps), Stories-110M to loss 3.40 (1B tokens)
- **Key findings:**
  - ANE = 24% of step time, CPU/AMX = 59%, IOSurface transfer = 9%, overhead = 8%
  - Pipeline overlap = 0% (measurement bug + design limitation — confirmed our similar issue)
  - **Activation explosion** over long runs: x grows to [-1850, 598] without `maxact` clipping
  - ANE compile budget is system-wide — rapid exec() loops poison the daemon
  - Cannot reduce ACCUM below ~100 with their 86-kernel pipeline (we have only 10 kernels — much better)
  - QoS 9 (Background) is fastest for ANE dispatch
  - Loss scaling of 1024 needed for FP16 underflow (we use 512 — could try higher)
  - `maxact` (activation clipping to [-100, 100]) stabilizes long training
- **Model sweep results:**
  - tiny_ane (13.3M): 26ms/step
  - small_ane (27.6M): 41ms/step
  - medium_ane (43.7M): 49.7ms/step
  - wide_ane (52.4M): 47.8ms/step
  - **All converge to loss ~5.9 in 200 steps** — larger models barely better at this scale

**Actionability:** ⭐⭐ MEDIUM
- **maxact / activation clipping:** We see stability issues too (especially with larger LR). Adding `GRAD_CLIP_MAX` already helps but per-element activation clipping could prevent the x explosion
- **QoS tuning:** We should verify we're using QoS 9 for ANE dispatch
- **Their pipeline overlap bug confirms ours** — the double-buffered overlap measurement is always 0 by construction. Good to know it's not just us.
- **Loss scale 1024:** Worth trying vs our current 512

### 3. Espresso — `christopherkarani/Espresso`

**What:** Pure Swift ANE runtime, now with autoresearch branch + Stories distillation experiments.
- 4.76x throughput vs CoreML (1.08ms/token vs 5.09ms/token)
- **New since March 27:** "Tahoe GPT-2 generation", "Stories future-head offline distillation", "fuse llama exact ANE head for Stories"
- **Key technique:** "donor delta" — using a teacher model to guide student training
- **New branch `autoresearch`:** merged from experiments branch

**Actionability:** ⭐ LOW for our pipeline (Swift, different architecture), but interesting for:
- Offline distillation techniques could improve training efficiency
- The "exact ANE head" fusion pattern matches our approach

### 4. ANE LoRA Training — `jmanhype/ane-lora-training`

**What:** First hybrid MLX + ANE system with fused LoRA gradient dispatch.
- Fused 4-matmul LoRA kernel (1 dispatch per module)
- Qwen2.5-3B-4bit LoRA: loss 4.07 → 2.64 in 200 steps at 155ms/step
- **Key insight:** MLX handles inference (GPU), ANE handles gradients — clean separation
- Python ctypes API to ANE bridge

**Actionability:** ⭐ LOW — LoRA-specific, but the fused kernel pattern is relevant

### 5. AutoANE — `vatsal191201/AutoANE`

**What:** Research into Zeroth-Order LoRA on ANE.
- Key finding: "ZO-LoRA Quality Ceiling" — gradient-free optimization on ANE has hard quality limits
- "Recursive Transformers + ANE: the natural architecture match"
- Paper draft on DSZO (Directional Stochastic Zeroth-Order)

**Actionability:** ⭐ LOW — confirms that gradient-based training (what we do) is essential for quality

### 6. Upstream `karpathy/autoresearch`

**New commits since last check:**
- `228791f` — bug fix in analysis.ipynb (best_bpb definition)
- `e6d79c1` — README enhancement
- `32a1460` — AMD ROCm fork link

**No new training techniques or architecture changes.**

### 7. Our `/Users/pavel/repos/fiale-plus/ANE` repo

**No new commits** since 20cd236 (INT8 bench). Same state as documented in 2026-03-28 knowledge source.

### 8. maderix/ANE PRs #39 and #40

**No updates** — both still open, last touched 2026-03-04.
- PR#39 (embedding 12x speedup) — still actionable, drop-in replacement
- PR#40 (E5 runtime) — still not actionable

## Critical ANE Constraints (Reaffirmed)

1. **GQA with N_KV_HEADS < HEADS crashes MIL compiler** — KEEP N_KV_HEADS=HEADS
2. **Matmul inner dim must be multiple of 32** — silent zeros on violation
3. **IOSurface slots must be ascending (input) / descending (output)** — silent zeros on violation
4. **~100 kernel compilations before process restart needed** (ANECompilerService leak) — we have 10 kernels, so ~10 recompiles = ~100 steps before issue
5. **Thermal throttle after ~60 min continuous** (+60% step time)
6. **512 channels only on M3 Ultra** — our DIM=768 works via tiling, not single-channel
7. **`constexpr_affine_dequantize` incompatible with dynamic pipeline** — INT8 forward not viable for us
8. **Mixed weight limit: ~3 FFN layers max per mega-kernel** — our fused kernels respect this

## Updated Integration Plan for Next Run

### Tier 1 — Config-only changes (no code changes, keep under program_ane.md rules)
1. **LOSS_SCALE 512 → 1024** — Slavko shows this stabilizes gradients
2. **ACCUM ramp experimentation** — our v5 cycle approach works, keep iterating
3. **Try LR_MIN_FRAC=0.01 → 0.05** — allow more aggressive decay
4. **GRAD_CLIP_MAX experimentation** — 1.0 vs 0.5 vs 2.0

### Tier 2 — Infrastructure upgrades (requires updating train_ane.m + adding files)
1. **Metal fused Lion kernel** — port Slavko's `adam_kernel.metal` pattern to Lion (simpler: no v buffer, sign-based update)
2. **Embedding 12x speedup** — port maderix PR#39 (cache-optimized gather + vDSP_mtrans)
3. **Activation clipping** — add maxact parameter to prevent x explosion in long runs
4. **QoS verification** — confirm ANE dispatches at QoS 9 (Background)

### Tier 3 — Research directions (future runs)
1. **Wider model sweep** — try DIM=512 (faster per step, more steps in 5 min) vs DIM=1024 (better per-step quality)
2. **Longer warmup** — try LR_WARMUP_STEPS=200, 300
3. **SWA (Stochastic Weight Averaging)** — average last N checkpoints for better generalization
4. **Different optimizers** — try AdaFactor, Shampoo (if implementable in FP16)

## Source Repositories Scanned

| Repo | Last Commit | Relevance | Key Takeaway |
|------|------------|-----------|--------------|
| slavko-at-klincov-it/ANE-Training | 2026-04-05 | ⭐⭐⭐ | Metal fused Adam, comprehensive benchmarks, libane C API |
| jmanhype/ane-lora-training | 2026-04-06 | ⭐ | Hybrid MLX+ANE, fused LoRA dispatch |
| christopherkarani/Espresso | 2026-04-07 | ⭐ | Swift ANE runtime, Stories distillation |
| vatsal191201/AutoANE | 2026-03-16 | ⭐ | ZO optimization ceiling paper |
| maderix/ANE | 2026-03-04 | ⭐⭐ | Embedding speedup (PR#39), E5 runtime (PR#40) |
| karpathy/autoresearch | 2026-03-26 | ⭐ | Bug fixes only, no new techniques |
| /Users/pavel/repos/fiale-plus/ANE | 2026-03-12 | ⭐⭐ | INT8 bench (not viable), multi-model infra |
