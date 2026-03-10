// experiment_config.h — Agent edits ONLY this file
// Architecture (changing these resets checkpoint)
#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define SEQ 256
#define NLAYERS 12
// Optimizer (safe to change between runs)
#define LEARNING_RATE 1e-3f
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.999f
#define ADAM_EPS 1e-8f
#define ACCUM_STEPS 4
#define GRAD_CLIP_MAX 1.0f
#define WEIGHT_DECAY 0.1f
#define WARMDOWN_RATIO 0.5f
// Experiment knobs to explore:
//   LR: try 8e-4, 9e-4, 1e-3, 1.1e-3, 1.2e-3 (best so far: 1e-3)
//   ACCUM_STEPS: try 1-2 for more weight updates per wall-time budget
//   HIDDEN: try 3072 (4x DIM) for wider FFN
//   SEQ: try 128 for faster steps (more iterations per budget)
//   WEIGHT_DECAY: CUDA uses 0.2, start at 0.1 (only on weight matrices, not embed/rmsnorm)
//   WARMDOWN_RATIO: fraction of wall-time for LR decay to 0 (CUDA uses 0.5)
//   ADAM_BETA1/BETA2: CUDA uses (0.8, 0.95) — try these values
