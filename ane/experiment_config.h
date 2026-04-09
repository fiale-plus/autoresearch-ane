// experiment_config.h — Agent edits ONLY this file
// Architecture (changing these resets checkpoint)
#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define N_KV_HEADS HEADS
#define SEQ 512
#define NLAYERS 6
// v13: Final definitive run — best config from v9/v12 experiments
// Key innovations: LOSS_SCALE=1024 (ecosystem), EMBED_LR_SCALE=1.0 (equal LR)
#define LEARNING_RATE 5e-4f
#define ADAM_BETA1 0.9f
#define ADAM_BETA2 0.95f
#define ADAM_EPS 1e-8f
#define ACCUM_STEPS 16
#define GRAD_CLIP_MAX 1.0f
#define WEIGHT_DECAY 0.1f
#define TOTAL_STEPS 3000
#define LR_WARMUP_STEPS 100
#define LR_MIN_FRAC 0.1f
#define LOSS_SCALE 1024.0f
#define SOFTCAP 30.0f
#define EMBED_LR_SCALE 1.0f
#define MATRIX_LR_SCALE 0.1f
#define USE_LION 1
#define USE_VOCAB_COMPACT 1