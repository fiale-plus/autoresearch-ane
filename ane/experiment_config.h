// experiment_config.h — Agent edits ONLY this file
// Architecture (changing these resets checkpoint)
#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define N_KV_HEADS HEADS   // GQA: set < HEADS for grouped-query attention (must divide HEADS evenly)
#define SEQ 512
#define NLAYERS 6
// Optimizer (safe to change between runs)
#define LEARNING_RATE 5e-4f
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.95f
#define ADAM_EPS 1e-8f
// Gradient accumulation steps before Adam update + weight re-staging (~50ms).
// Lower = more weight updates per wall-clock minute. With dynamic pipeline,
// there is no recompilation cost — only the cheap weight transpose + IOSurface write.
#define ACCUM_STEPS 12
#define GRAD_CLIP_MAX 1.0f
#define WEIGHT_DECAY 0.1f
// LR schedule: cosine decay with linear warmup
// Total steps for cosine LR schedule denominator (adam_t, not forward steps).
// Best val_loss at adam_t≈1600. Set to ~3K for proper decay across 5-cycle ramp.
#define TOTAL_STEPS 3000
#define LR_WARMUP_STEPS 100
#define LR_MIN_FRAC 0.1f
// Loss scaling: prevents FP16 gradient underflow (maderix/ncdrone both use this)
#define LOSS_SCALE 512.0f
// Logit softcapping: cap * tanh(logits/cap), prevents logit explosion
#define SOFTCAP 30.0f
// Differential learning rates (ncdrone: embed 5x faster, matrices 20x slower)
#define EMBED_LR_SCALE 2.0f
#define MATRIX_LR_SCALE 0.1f
// === Pipeline features ===
// Lion optimizer: sign-based updates, no second moment (v buffer), ~2x faster per update.
// Recommended: LR ~3x lower than Adam, WD ~3x higher. Resets checkpoint: YES (different optimizer state)
#define USE_LION 1
// Vocab compaction: only ~9K of 32K tokens appear in TinyStories. Reduces classifier SGEMM ~3.5x.
// No accuracy impact — unused tokens are never trained. Resets checkpoint: NO
#define USE_VOCAB_COMPACT 1
