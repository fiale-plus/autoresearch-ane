# Knowledge Sources Update — 2026-03-28

## Source: `ane-techniques-v2` branch (this repo)

**GQA (Grouped-Query Attention) support** — commit 0a058d5
- New parameter: `N_KV_HEADS` (default = HEADS for MHA backward compat)
- KV_DIM = N_KV_HEADS × HEAD_DIM — enables wider/deeper models within same param budget
- MIL kernels adapted: split SDPA forward into sdpaFwd + woFwd, separate qBwd/kvBwd
- Tile-based KV head expansion in forward pass (broadcast KV→HEADS)
- `reduce_sum` for gradient accumulation back to KV_DIM in backward
- IOSurface staging adjusted for asymmetric Q/KV dimensions (WK_SZ/WV_SZ)
- Checkpoint v4: `n_kv_heads` field with backward compat (0 → n_heads)

**Codebase cleanup**
- Removed broken `USE_RMSNORM_FUSION` toggle — ANE rejects rsqrt/broadcast in channel dim
- Removed `USE_INT8_ACTIVATIONS` toggle — experimental, convergence issues
- Removed `USE_CHANNEL_FIRST_CPU` toggle — always-on now, weights pre-transposed, matmul uses transpose_y=bF
- Simplified `gen_dyn_matmul_mil` — removed transpose_wt parameter
- Net -147 lines across MIL kernels

## Source: `/Users/pavel/repos/fiale-plus/ANE` repo

**INT8 W8A8 quantization benchmark** — commit 20cd236
- M4 results: FP16 18.6 TOPS → INT8 W8A8 35.1 TOPS (1.88× speedup)
- 128×512ch conv on 64×64 input: 14.8ms FP16 → 7.8ms INT8
- Implementation: `constexpr_affine_dequantize` for weights, MIL `quantize`/`dequantize` for activations
- Standalone benchmark in `ane_int8_bench.m`
- Not yet integrated into training pipeline (INT8 backward pass is the missing piece)

**Multi-model architecture** — commits 475348a, 7d61ee4
- Models configured via `models/*.h` headers, selected at build: `make MODEL=qwen3_06b`
- Q_DIM separated from KV_DIM for proper GQA
- Per-layer IOSurfaces + requests structs for cleaner resource management
- CkptHdr v4 with `kv_heads`, `head_dim`, `q_dim` fields
- `arm_neon.h` NEON SIMD includes
- Models verified: Stories110M (12L/768d/MHA ~91-115ms/step), Qwen3-0.6B (28L/1024d/GQA 16/8 ~412ms/step)

**Dashboard & infra** — commit 7d61ee4
- Live training stats, W&B integration (--wandb flag)
- GQA-aware text generation with KV cache
- Top-k=50 sampling, variable vocab tokenizer
- Best-loss checkpointing

## Source: upstream `karpathy/autoresearch`

- Bug fixes only (analysis.ipynb best_bpb, README enhancements). No new training techniques.

## Source: `origin/updates/knowledge-sources-2026-03-28` (remote branch — ecosystem scan)

**Embedding lookup 12x speedup** — maderix/ANE PR #39
- Cache-optimized: contiguous `memcpy` gather + `vDSP_mtrans` transpose
- Eliminates stride-seq cache misses in embed_lookup
- 0.39ms → 0.033ms per call on M4 Max
- Drop-in replacement — actionable for train_ane.m

**E5 Runtime** — maderix/ANE PR #40
- `_ANEChainingRequest` API dead on macOS 15+
- Modern path: `MLE5Engine` via `MLE5ProgramLibraryOnDeviceAOTCompilationImpl`
- 7 test programs, ~7K lines of reverse-engineering
- Not actionable yet (major rewrite)

**New repos discovered**
- imperatormk/ane-train — runtime IOSurface weight injection, 28-block ConvNeXt UNet ~3 it/s M1
- christopherkarani/Espresso — pure Swift, 4.76x throughput vs CoreML (1.08ms/token vs 5.09ms)
- ncdrone/rustane — Rust-native hybrid ANE + Metal GPU training

**Critical ANE hardware constraints** (compiled from ecosystem)
- **GQA with non-equal KV heads crashes MIL compiler** — must keep N_KV_HEADS=HEADS for now!
- Mixed weight limit: 16 total weights minus n_norms per mega-kernel (~3 FFN layers max)
- IOSurface slots must be ascending (input) / descending (output) — silent zeros on violation
- Matmul inner dim must be multiple of 32 — silent zeros on violation
- `constexpr_affine_dequantize` incompatible with dynamic pipeline (bakes weights at compile time)
- Thermal throttle after ~60 min continuous (+60% step time)
- ~100 kernel compilations before process restart needed (ANECompilerService leak)
- 512 channels ONLY valid on M3 Ultra (all others fail with -4/-3)

## Integration plan for mar28 (revised)

1. **Branch from `ane-techniques-v2`** — gets GQA code + clean kernels ✅
2. **DO NOT experiment with N_KV_HEADS < HEADS** — crashes MIL compiler per ecosystem findings
3. **Integrate embedding 12x speedup** — drop-in replacement in train_ane.m (cache-optimized gather)
4. **Hyperparameter sweep** — ACCUM_STEPS, LR, architecture (DIM, HIDDEN, NLAYERS) — MHA only
5. **INT8 forward-only** — not viable (incompatible with dynamic pipeline), skip
6. **Watch for thermal throttle** — monitor step time in long runs, may need cooling breaks
