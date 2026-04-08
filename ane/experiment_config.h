// experiment_config.h — Agent edits ONLY this file
// Architecture (changing these resets checkpoint)
#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define N_KV_HEADS HEADS   // GQA: MUST stay HEADS (MIL compiler crashes otherwise)
#define SEQ 512
#define NLAYERS 6
// Optimizer (safe to change between runs)
#define LEARNING_RATE 5e-4f
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.95f
#define ADAM_EPS 1e-8f
// v7 experiment: extended ramp with LOSS_SCALE=1024 from ecosystem research
#define ACCUM_STEPS 4
#define GRAD_CLIP_MAX 1.0f
#define WEIGHT_DECAY 0.1f
// LR schedule: cosine decay with linear warmup
#define TOTAL_STEPS 3000
#define LR_WARMUP_STEPS 100
#define LR_MIN_FRAC 0.1f
// Loss scaling: ecosystem (Slavko) shows 1024 is better for FP16
#define LOSS_SCALE 1024.0f
// Logit softcapping: cap * tanh(logits/cap), prevents logit explosion
#define SOFTCAP 30.0f
// Differential learning rates
#define EMBED_LR_SCALE 2.0f
#define MATRIX_LR_SCALE 0.1f
// === Pipeline features ===
#define USE_LION 1
#define USE_VOCAB_COMPACT 1
