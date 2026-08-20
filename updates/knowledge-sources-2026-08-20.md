# Knowledge Sources Update — 2026-08-20

## Repository and host sync state

- Fresh local probe ran on Apple `Mac15,6`, macOS `26.5.1` (build `25F80`), 36 GB unified memory, 12 logical CPUs.
- The retained local best remains **val_loss 2.320954** from the 2026-05-26 checkpoint trajectory.
- Three single-variable probes were run from a preserved copy of that checkpoint, each with `ANE_WALL_TIME=300` (about 4.9 minutes of training):

| Probe | val_loss | ms/step | ANE util | Verdict |
|---|---:|---:|---:|---|
| `ACCUM_STEPS=1` | 2.401719 | 137.7 | 4.5% | discard |
| `LEARNING_RATE=3.6e-4f`, `ACCUM_STEPS=2` | 2.346754 | 98.4 | 6.4% | discard |
| `LEARNING_RATE=3.7e-4f`, `ACCUM_STEPS=2` | 2.348575 | 97.0 | 6.5% | discard |

The committed configuration is unchanged: Lion, `LEARNING_RATE=3.8e-4f`, `ACCUM_STEPS=2`, `WEIGHT_DECAY=0.1f`, `LOSS_SCALE=1024`, `EMBED_LR_SCALE=1.0`, and `MATRIX_LR_SCALE=0.1`.

## New ecosystem findings

### 1. `ncdrone/rustane` — dispatch and shape discipline

The current README reports validated training from 579M to 5B parameters and forward-only probes to 30B on an M4 Max 128 GB. Its most actionable result for this repository is architectural shape discipline: wide+shallow configurations win below about 3B parameters, while deep+narrow wins at larger scales; `dim=5120` is an efficiency cliff and dimensions at or below 4096 are recommended.

**Actionability:** high as a design constraint, low as a drop-in dependency. The current `DIM=768`, `HIDDEN=2048`, and `SEQ=512` remain safely below the reported cliff class. Future architecture probes should measure dispatch count and step time rather than infer quality from parameter count alone.

Source: <https://github.com/ncdrone/rustane>

### 2. `jmanhype/ane-lora-training` — fused dynamic dispatch

The current README reports a packed-IOSurface dynamic matmul that compiles once and a fused four-matmul LoRA gradient kernel reducing four dispatches to one. It also documents that ANE spatial dimensions must be at least 16 and multiples of 16, including output dimensions.

**Actionability:** high for future infrastructure work. The current backend already uses dynamic IOSurface weights and fused forward/backward kernels; the next code-level breakthrough should target remaining dispatch boundaries or optimizer/gradient fusion, not another broad config sweep.

Source: <https://github.com/jmanhype/ane-lora-training>

### 3. `slavko-at-klincov-it/ANE-Training` — CPU/ANE split and compile-once execution

The current reference continues to report a compile-once dynamic spatial-packing design and a measured Stories-110M training split of roughly 24% ANE, 59% CPU/AMX, 9% IOSurface transfer, and 8% other overhead. Its README describes fused optimizer patterns and explicitly frames ANE training as an ANE+CPU pipeline rather than pure ANE execution.

**Actionability:** high. The local runs show only 6.4–6.5% reported ANE utilization, so reducing CPU-side gradient/optimizer overhead and dispatch boundaries is a more credible next direction than changing the model size.

Source: <https://github.com/slavko-at-klincov-it/ANE-Training>

### 4. `maderix/ANE` — current baseline limits and INT8 lead

The current README reports 91 ms/step for a dynamic 109M Stories model and 412 ms/step for a dynamic 596M GQA model, with CPU dW gradients and optimizer work. It also reports an INT8 W8A8 benchmark at about 1.85–1.88x FP16 throughput on a microbenchmark, while warning that the project is research code and private APIs are unstable.

**Actionability:** medium. INT8 is a possible future throughput experiment, but it is not a config-only change and needs a correctness/gradient validation path before entering this autoresearch loop.

Source: <https://github.com/maderix/ANE>

## Apple platform status

Apple’s WWDC26 machine-learning guide now positions **MLX** as the supported path for Apple-silicon research, training, and fine-tuning, with Metal 4 and GPU Neural Accelerator support. It positions **Core AI** as the supported path for loading, specializing, compiling, and running models on-device; the guide does not document general-purpose backpropagation or direct arbitrary ANE training through Core AI. This does not replace the private-API research value of this repository, but it confirms that no public macOS 26 API removes the need for the current reverse-engineered backend.

The host is on macOS 26.5.1. No OS-specific code change was justified by this scan; the current private API path should continue to be treated as version-sensitive and research-only.

Sources:

- <https://developer.apple.com/wwdc26/guides/machine-learning/>
- <https://developer.apple.com/documentation/CoreAI>
- <https://developer.apple.com/documentation/metal/machine-learning-passes>

## Updated next-action plan

1. Keep the current best config as the sticky baseline; the August probes did not beat it.
2. Prefer code-level profiling and dispatch-count reduction over nearby accumulation/LR sweeps.
3. Investigate fused Lion/gradient updates and embedding lookup/packing only with a reproducible correctness check and a short timing probe before a full five-minute run.
4. Treat INT8 and architecture changes as separate experiments requiring checkpoint resets and explicit numerical validation.
