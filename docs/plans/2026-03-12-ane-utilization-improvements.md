# ANE Utilization Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Increase ANE utilization from 3.6% by adding three pipeline optimizations as agent-toggleable experiment_config.h defines.

**Architecture:** Three independent #if-guarded features in the ANE training pipeline: (1) eliminate CPU transpose overhead for square weight matrices, (2) fuse RMSNorm into ANE MIL kernels, (3) add INT8 activation quantization between MIL ops. Each feature is a separate commit. All changes are compile-time — toggling a define requires `make clean && make`.

**Tech Stack:** Objective-C, MIL (Apple Neural Engine IR), IOSurface, vDSP/NEON, cblas

---

## File Map

| File | Role |
|------|------|
| `ane/experiment_config.h` | Agent-editable defines (architecture + optimizer + NEW feature toggles) |
| `ane/stories_mil_dynamic.h` | MIL kernel generators (10 kernels). Changes for all 3 features. |
| `ane/stories_io.h` | IOSurface staging, SP size macros, blob builders. Changes for features 1+2. |
| `ane/train_ane.m` | Training loop. Changes for features 1+2. |
| `program_ane.md` | Agent instructions. Document new toggles. |

## Key Constraints

- ANE single-input constraint: all data (activations + weights) packed into one IOSurface spatial dimension
- Weights are [1, CH, 1, SP] in IOSurface — CH channels, SP spatial positions per channel
- Forward kernels need W^T (transposed weight); backward kernels need W (original)
- Square weights (Wq,Wk,Wv,Wo) are [DIM,DIM]; rectangular (W1,W3) are [HIDDEN,DIM]
- MIL `matmul(transpose_y=bT)` transposes y before multiply — free on ANE (view change)
- `concat` on axis=1 works in our kernels (despite orion constraint #1 — verified by existing code)
- Output IOSurface is per-kernel (shared across layers), overwritten each eval

---

### Task 1: Feature toggles in experiment_config.h

**Files:**
- Modify: `ane/experiment_config.h`

**Step 1: Add feature defines after existing optimizer defines**

Add at end of file:

```c
// === ANE pipeline features (toggle to experiment) ===
// Channel-first square weights: stage Wq,Wk,Wv,Wo directly without CPU transpose.
// Saves ~75MB RAM and 32 vDSP_mtrans calls per weight restage. Backward kernels unaffected.
// Resets checkpoint: NO (same weights, different staging path)
#define USE_CHANNEL_FIRST_CPU 1
// RMSNorm fusion: compute RMSNorm inside sdpaFwd and ffnFused ANE kernels instead of CPU.
// Eliminates 16 CPU RMSNorm calls per step (2 per layer). Outputs xnorm/x2norm for backward.
// Resets checkpoint: NO (numerically equivalent in fp16)
#define USE_RMSNORM_FUSION 1
// INT8 activation quantization: add quantize/dequantize between matmul ops in forward MIL kernels.
// Reduces ANE L2 SRAM bandwidth ~2x between tiles. May affect convergence — experiment!
// Resets checkpoint: NO (activations only, weights stay fp16/fp32)
#define USE_INT8_ACTIVATIONS 0
```

**Step 2: Compile to verify no breakage**

Run: `make -C ane clean && make -C ane train_ane`
Expected: compiles successfully (defines unused so far)

**Step 3: Commit**

```bash
git add ane/experiment_config.h
git commit -m "feat: add ANE pipeline feature toggles to experiment_config.h"
```

---

### Task 2: Channel-first square weights — MIL changes

**Files:**
- Modify: `ane/stories_mil_dynamic.h`

**Context:** gen_dyn_matmul_mil is shared by forward (woFwd) and backward (ffnBwdW2t, wotBwd, qBwd) kernels. Forward needs `transpose_y=bT` when staging original weights; backward keeps `transpose_y=bF`.

**Step 1: Add transpose_wt parameter to gen_dyn_matmul_mil**

Change the function signature and the matmul line:

```c
static NSString *gen_dyn_matmul_mil(int ic, int oc, int seq, int transpose_wt) {
```

Change the matmul line from:
```c
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n", seq, oc];
```
To:
```c
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=%@,x=a3,y=W)[name=string(\"yh\")];\n",
        transpose_wt ? oc : seq, transpose_wt ? ic : oc,
        transpose_wt ? @"bT" : @"bF"];
```

Wait — with transpose_y=bT, the output shape changes: act [SEQ,IC] @ W^T where W is [IC,OC] → W^T is [OC,IC] → matmul result is [SEQ,IC]×[IC,OC] ... No. With transpose_y, matmul treats y as y^T. So if y=[IC,OC], then y^T=[OC,IC]. The matmul is x[SEQ,IC] @ y^T[OC,IC]... that's incompatible.

Let me reconsider. When transpose_wt=1, the STAGED data represents W_original [IC,IC] (for square). In the kernel, after reshape it's [1,1,IC,OC] where OC=IC. With transpose_y=bT: act[SEQ,IC] @ y^T = act[SEQ,IC] @ [OC,IC]^T = act[SEQ,IC] @ [IC,OC] = [SEQ,OC]. ✓

For non-square when transpose_wt=1: the staged data represents W_original. But the reshape is to [1,1,IC,OC] where the data IS W_original[IC,OC] (since we staged rows). With transpose_y=bT: act[SEQ,IC] @ W_orig[IC,OC]^T = act[SEQ,IC] @ [OC,IC] ... dimension mismatch unless IC matches.

**Critical insight:** transpose_wt ONLY works for square matrices (IC=OC=DIM) in gen_dyn_matmul_mil. For rectangular, keep the existing approach.

So the matmul output shape stays [SEQ, OC] in both cases (for square IC=OC):

```c
static NSString *gen_dyn_matmul_mil(int ic, int oc, int seq, int transpose_wt) {
    // ... existing code unchanged until matmul line ...
    // When transpose_wt: staged weight is W_original [IC,IC], transpose_y=bT computes act @ W^T
    // When !transpose_wt: staged weight is W^T [IC,OC], transpose_y=bF computes act @ W^T
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> yh = matmul(transpose_x=bF,transpose_y=%@,x=a3,y=W)[name=string(\"yh\")];\n",
        seq, oc, transpose_wt ? @"bT" : @"bF"];
```

**Step 2: Update all callers of gen_dyn_matmul_mil**

```c
// woFwd: forward square weight
static NSString *gen_wo_fwd_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, DIM, SEQ, USE_CHANNEL_FIRST_CPU);
}

// ffnBwdW2t: backward rectangular weight — no change
static NSString *gen_ffn_bwd_w2t_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, HIDDEN, SEQ, 0);
}

// wotBwd: backward square weight — no change (already stages original)
static NSString *gen_wot_bwd_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, DIM, SEQ, 0);
}

// qBwd: backward square weight — no change
static NSString *gen_q_bwd_dynamic(void) {
    return gen_dyn_matmul_mil(DIM, DIM, SEQ, 0);
}
```

**Step 3: Update gen_sdpa_fwd_dynamic QKV matmuls**

The 3 matmul lines (Q, K, V projections) change transpose_y conditionally:

```c
#if USE_CHANNEL_FIRST_CPU
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> qm = matmul(transpose_x=bF,transpose_y=bT,x=xnt,y=Wq2)[name=string(\"qm\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> km = matmul(transpose_x=bF,transpose_y=bT,x=xnt,y=Wk2)[name=string(\"km\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> vm = matmul(transpose_x=bF,transpose_y=bT,x=xnt,y=Wv2)[name=string(\"vm\")];\n", SEQ, DIM];
#else
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> qm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wq2)[name=string(\"qm\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> km = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wk2)[name=string(\"km\")];\n", SEQ, DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> vm = matmul(transpose_x=bF,transpose_y=bF,x=xnt,y=Wv2)[name=string(\"vm\")];\n", SEQ, DIM];
#endif
```

---

### Task 3: Channel-first square weights — staging + train_ane.m

**Files:**
- Modify: `ane/stories_io.h` (staging function signatures — conceptual only, actual code identical)
- Modify: `ane/train_ane.m` (eliminate Wqt_buf, change staging calls)

**Step 1: In train_ane.m, conditionally eliminate Wqt_buf arrays**

Around line 317-329, wrap the transpose buffer allocation and initial transpose:

```c
#if !USE_CHANNEL_FIRST_CPU
        float *Wqt_buf[NLAYERS], *Wkt_buf[NLAYERS], *Wvt_buf[NLAYERS], *Wot_buf[NLAYERS];
#endif
        float *W1t_buf[NLAYERS], *W3t_buf[NLAYERS];
        for (int L=0; L<NLAYERS; L++) {
#if !USE_CHANNEL_FIRST_CPU
            Wqt_buf[L]=(float*)malloc(WQ_SZ*4); Wkt_buf[L]=(float*)malloc(WQ_SZ*4);
            Wvt_buf[L]=(float*)malloc(WQ_SZ*4); Wot_buf[L]=(float*)malloc(WO_SZ*4);
            transpose_weight(Wqt_buf[L], lw[L].Wq, DIM, DIM);
            transpose_weight(Wkt_buf[L], lw[L].Wk, DIM, DIM);
            transpose_weight(Wvt_buf[L], lw[L].Wv, DIM, DIM);
            transpose_weight(Wot_buf[L], lw[L].Wo, DIM, DIM);
#endif
            W1t_buf[L]=(float*)malloc(W1_SZ*4); W3t_buf[L]=(float*)malloc(W3_SZ*4);
            transpose_weight(W1t_buf[L], lw[L].W1, HIDDEN, DIM);
            transpose_weight(W3t_buf[L], lw[L].W3, HIDDEN, DIM);
        }
```

**Step 2: Change initial weight staging (line 365-374)**

```c
        for (int L = 0; L < NLAYERS; L++) {
#if USE_CHANNEL_FIRST_CPU
            stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, lw[L].Wq, lw[L].Wk, lw[L].Wv);
            stage_wo_fwd_weights(pls[L].woFwd_in, lw[L].Wo);
#else
            stage_sdpa_fwd_weights(pls[L].sdpaFwd_in, Wqt_buf[L], Wkt_buf[L], Wvt_buf[L]);
            stage_wo_fwd_weights(pls[L].woFwd_in, Wot_buf[L]);
#endif
            stage_ffn_fused_weights(pls[L].ffnFused_in, W1t_buf[L], W3t_buf[L], lw[L].W2);
            // backward staging unchanged — already uses original weights
            stage_ffn_bwd_w2t_weights(pls[L].ffnBwdW2t_in, lw[L].W2);
            stage_ffn_bwd_w13t_weights(pls[L].ffnBwdW13t_in, lw[L].W1, lw[L].W3);
            stage_wot_bwd_weights(pls[L].wotBwd_in, lw[L].Wo);
            stage_q_bwd_weights(pls[L].qBwd_in, lw[L].Wq);
            stage_kv_bwd_weights(pls[L].kvBwd_in, lw[L].Wk, lw[L].Wv);
        }
```

**Step 3: Change restage inside dispatch_apply (line 714-729)**

Replace the transpose + restage block:

```c
                    // Update transposed weight buffers and re-stage
#if USE_CHANNEL_FIRST_CPU
                    // Square weights: stage original directly (MIL uses transpose_y=bT)
                    stage_sdpa_fwd_weights(p_pls[L].sdpaFwd_in, p_lw[L].Wq, p_lw[L].Wk, p_lw[L].Wv);
                    stage_wo_fwd_weights(p_pls[L].woFwd_in, p_lw[L].Wo);
#else
                    transpose_weight(p_Wqt[L], p_lw[L].Wq, DIM, DIM);
                    transpose_weight(p_Wkt[L], p_lw[L].Wk, DIM, DIM);
                    transpose_weight(p_Wvt[L], p_lw[L].Wv, DIM, DIM);
                    transpose_weight(p_Wot[L], p_lw[L].Wo, DIM, DIM);
                    stage_sdpa_fwd_weights(p_pls[L].sdpaFwd_in, p_Wqt[L], p_Wkt[L], p_Wvt[L]);
                    stage_wo_fwd_weights(p_pls[L].woFwd_in, p_Wot[L]);
#endif
                    // Rectangular weights: still need transpose (strided gather not faster)
                    transpose_weight(p_W1t[L], p_lw[L].W1, HIDDEN, DIM);
                    transpose_weight(p_W3t[L], p_lw[L].W3, HIDDEN, DIM);
                    stage_ffn_fused_weights(p_pls[L].ffnFused_in, p_W1t[L], p_W3t[L], p_lw[L].W2);
                    // backward restaging unchanged
                    stage_ffn_bwd_w2t_weights(p_pls[L].ffnBwdW2t_in, p_lw[L].W2);
                    stage_ffn_bwd_w13t_weights(p_pls[L].ffnBwdW13t_in, p_lw[L].W1, p_lw[L].W3);
                    stage_wot_bwd_weights(p_pls[L].wotBwd_in, p_lw[L].Wo);
                    stage_q_bwd_weights(p_pls[L].qBwd_in, p_lw[L].Wq);
                    stage_kv_bwd_weights(p_pls[L].kvBwd_in, p_lw[L].Wk, p_lw[L].Wv);
```

**Step 4: Guard the pointer captures and cleanup for Wqt/Wkt/Wvt/Wot**

Line 697-698 (pointer captures):
```c
#if !USE_CHANNEL_FIRST_CPU
                float **p_Wqt = Wqt_buf, **p_Wkt = Wkt_buf, **p_Wvt = Wvt_buf, **p_Wot = Wot_buf;
#endif
```

Line 829 (cleanup):
```c
#if !USE_CHANNEL_FIRST_CPU
            free(Wqt_buf[L]); free(Wkt_buf[L]); free(Wvt_buf[L]); free(Wot_buf[L]);
#endif
            free(W1t_buf[L]); free(W3t_buf[L]);
```

**Step 5: Compile and smoke-test**

Run: `make -C ane clean && make -C ane train_ane && ANE_WALL_TIME=30 python harness_ane.py`
Expected: compiles, runs 30s without crash, produces val_loss

**Step 6: Commit**

```bash
git add ane/stories_mil_dynamic.h ane/train_ane.m
git commit -m "feat: channel-first square weights — eliminate transpose for Wq/Wk/Wv/Wo staging"
```

---

### Task 4: RMSNorm fusion — MIL kernel changes

**Files:**
- Modify: `ane/stories_mil_dynamic.h`
- Modify: `ane/stories_io.h`

**Context:** Fuse RMSNorm into sdpaFwd and ffnFused kernels. Each kernel gets raw activation + gamma weight, computes norm internally, outputs the normed activation for backward.

**Step 1: Update SP macros in stories_io.h**

The spatial layout changes when RMSNorm is fused:

For sdpaFwd:
- Old: SEQ + 3*DIM (xnorm + Wq^T + Wk^T + Wv^T)
- New: SEQ + 3*DIM + 1 (x_raw + Wq + Wk + Wv + gamma_att[1 per channel])
- Size difference: +1 spatial position

For ffnFused:
- Old: 2*SEQ + 3*HIDDEN (x2norm + x2 + W1^T + W3^T + W2)
- New: SEQ + 3*HIDDEN + 1 (x2 + W1^T + W3^T + W2 + gamma_ffn[1 per channel])
- Size difference: -SEQ+1 (we DROP the x2norm copy since we compute it, and drop the separate x2 since we read it once and use for both norm and residual)

Output size changes:
- sdpaFwd: 4*DIM*SEQ → 5*DIM*SEQ (add xnorm output)
- ffnFused: (DIM+3*HIDDEN)*SEQ → (2*DIM+3*HIDDEN)*SEQ (add x2norm output)

```c
// In stories_io.h, replace the SP macros:
#if USE_RMSNORM_FUSION
#define SDPA_FWD_SP    (SEQ + 3*DIM + 1)     // x_raw + Wq + Wk + Wv + gamma_att
#define FFN_FUSED_SP   (SEQ + 3*HIDDEN + 1)  // x2 + W1^T + W3^T + W2 + gamma_ffn
#define SDPA_FWD_OUT_CH (5*DIM)               // attn_out + Q + K + V + xnorm
#define FFN_FUSED_OUT_CH (2*DIM + 3*HIDDEN)   // x_next + h1 + h3 + silu + x2norm
#else
#define SDPA_FWD_SP    (SEQ + 3*DIM)
#define FFN_FUSED_SP   (2*SEQ + 3*HIDDEN)
#define SDPA_FWD_OUT_CH (4*DIM)
#define FFN_FUSED_OUT_CH (DIM + 3*HIDDEN)
#endif
```

**Step 2: Add gamma staging and updated write functions in stories_io.h**

```c
#if USE_RMSNORM_FUSION
// Stage RMSNorm gamma into last spatial position of sdpaFwd surface
static void stage_sdpa_fwd_gamma(IOSurfaceRef s, const float *gamma_att) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    int gpos = SEQ + 3*DIM;  // last spatial position
    for (int d = 0; d < DIM; d++)
        buf[d * SDPA_FWD_SP + gpos] = (_Float16)gamma_att[d];
    IOSurfaceUnlock(s, 0, NULL);
}

// Stage RMSNorm gamma into last spatial position of ffnFused surface
static void stage_ffn_fused_gamma(IOSurfaceRef s, const float *gamma_ffn) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    int gpos = SEQ + 3*HIDDEN;
    for (int d = 0; d < DIM; d++)
        buf[d * FFN_FUSED_SP + gpos] = (_Float16)gamma_ffn[d];
    IOSurfaceUnlock(s, 0, NULL);
}

// Write raw x_cur (un-normalized) to sdpaFwd input — replaces xnorm write
static void write_sdpa_fwd_acts_raw(IOSurfaceRef s, const float *x_raw) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*SDPA_FWD_SP, x_raw + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// Write raw x2 to ffnFused input (single copy, used for both norm and residual)
static void write_ffn_fused_acts_raw(IOSurfaceRef s, const float *x2) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++)
        cvt_f32_f16(buf + d*FFN_FUSED_SP, x2 + d*SEQ, SEQ);
    IOSurfaceUnlock(s, 0, NULL);
}

// Updated FFN weight staging for fused layout (x2 at 0, weights at SEQ)
static void stage_ffn_fused_weights_fused(IOSurfaceRef s,
                                           const float *W1t, const float *W3t, const float *W2_orig) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *buf = (_Float16*)IOSurfaceGetBaseAddress(s);
    for (int d = 0; d < DIM; d++) {
        cvt_f32_f16(buf + d*FFN_FUSED_SP + SEQ,              W1t + d*HIDDEN, HIDDEN);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + SEQ+HIDDEN,       W3t + d*HIDDEN, HIDDEN);
        cvt_f32_f16(buf + d*FFN_FUSED_SP + SEQ+2*HIDDEN,     W2_orig + d*HIDDEN, HIDDEN);
    }
    IOSurfaceUnlock(s, 0, NULL);
}
#endif
```

**Step 3: Add RMSNorm MIL ops to gen_sdpa_fwd_dynamic in stories_mil_dynamic.h**

Insert after the input declaration and before the existing xnorm slice. When USE_RMSNORM_FUSION, compute RMSNorm in MIL:

```c
#if USE_RMSNORM_FUSION
    // Slice raw x [1,DIM,1,SEQ]
    [m appendString:@"        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xraw = slice_by_size(x=x,begin=bx,size=sx)[name=string(\"xraw\")];\n", DIM, SEQ];

    // Slice gamma [1,DIM,1,1]
    [m appendFormat:@"        tensor<int32, [4]> bg = const()[name=string(\"bg\"), val=tensor<int32, [4]>([0,0,0,%d])];\n", SEQ+3*DIM];
    [m appendFormat:@"        tensor<int32, [4]> sg = const()[name=string(\"sg\"), val=tensor<int32, [4]>([1,%d,1,1])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> gamma = slice_by_size(x=x,begin=bg,size=sg)[name=string(\"gamma\")];\n", DIM];

    // RMSNorm: xnorm = x * rsqrt(mean(x^2) + eps) * gamma
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xsq = mul(x=xraw,y=xraw)[name=string(\"xsq\")];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> xss = reduce_sum(x=xsq,axes=rax,keep_dims=kd)[name=string(\"xss\")];\n", SEQ];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%g)];\n", 1.0f/DIM];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> xms = mul(x=xss,y=invd)[name=string(\"xms\")];\n", SEQ];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(1e-5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> xme = add(x=xms,y=eps)[name=string(\"xme\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> irms = rsqrt(x=xme)[name=string(\"irms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn0 = mul(x=xraw,y=irms)[name=string(\"xn0\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xn0,y=gamma)[name=string(\"xn\")];\n", DIM, SEQ];
#else
    // Slice pre-computed xnorm [1,DIM,1,SEQ]
    [m appendString:@"        tensor<int32, [4]> bx = const()[name=string(\"bx\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sx = const()[name=string(\"sx\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = slice_by_size(x=x,begin=bx,size=sx)[name=string(\"xn\")];\n", DIM, SEQ];
#endif
```

Then update the output concat to include xnorm:

```c
#if USE_RMSNORM_FUSION
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(af,qf,kf,vf,xn))[name=string(\"cat\")];\n", 5*DIM, SEQ];
#else
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(af,qf,kf,vf))[name=string(\"cat\")];\n", 4*DIM, SEQ];
#endif
```

**Step 4: Add RMSNorm to gen_ffn_fused_dynamic**

Similar pattern. Input layout changes: x2 at spatial 0, weights at SEQ (not 2*SEQ). Compute RMSNorm on x2 to get x2norm. Use x2 for residual. Output adds x2norm.

Replace the input slicing section with conditionally compiled versions. The key MIL ops are the same reduce_sum → mul → add → rsqrt → mul → mul pattern.

Output concat becomes:
```c
#if USE_RMSNORM_FUSION
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(x_next,h1,h3,gate,x2norm))[name=string(\"cat\")];\n", 2*DIM+3*HIDDEN, SEQ];
#else
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(x_next,h1,h3,gate))[name=string(\"cat\")];\n", DIM+3*HIDDEN, SEQ];
#endif
```

---

### Task 5: RMSNorm fusion — train_ane.m changes

**Files:**
- Modify: `ane/train_ane.m`

**Step 1: Update kernel compilation output sizes (line 54-70)**

```c
    dk->sdpaFwd = compile_kern_mil_w(gen_sdpa_fwd_dynamic(), mask_w,
        DIM*SDPA_FWD_SP*2, SDPA_FWD_OUT_CH*SEQ*2);
    // ...
    dk->ffnFused = compile_kern_mil_w(gen_ffn_fused_dynamic(), @{},
        DIM*FFN_FUSED_SP*2, FFN_FUSED_OUT_CH*SEQ*2);
```

**Step 2: Stage gamma weights after regular weight staging (line 365-375)**

```c
#if USE_RMSNORM_FUSION
            stage_sdpa_fwd_gamma(pls[L].sdpaFwd_in, lw[L].rms_att);
            stage_ffn_fused_gamma(pls[L].ffnFused_in, lw[L].rms_ffn);
#endif
```

**Step 3: Update forward loop (line 435-482)**

Replace CPU RMSNorm + write with raw write when fused:

```c
                // CPU RMSNorm1 + write to ANE
#if USE_RMSNORM_FUSION
                memcpy(ac->layer_in, x_cur, SEQ*DIM*4);
                write_sdpa_fwd_acts_raw(pls[L].sdpaFwd_in, x_cur);
#else
                rmsnorm(xnorm_buf, x_cur, lw[L].rms_att, DIM, SEQ);
                memcpy(ac->xnorm, xnorm_buf, SEQ*DIM*4);
                memcpy(ac->layer_in, x_cur, SEQ*DIM*4);
                write_sdpa_fwd_acts(pls[L].sdpaFwd_in, xnorm_buf);
#endif
```

Read xnorm from SDPA output when fused:

```c
                // Read SDPA output
                IOSurfaceLock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *fwd_out = (_Float16*)IOSurfaceGetBaseAddress(dk.sdpaFwd->ioOut);
                int off = 0;
                cvt_f16_f32(ac->attn_out, fwd_out + off, DIM*SEQ); off += DIM*SEQ;
                cvt_f16_f32(ac->Q,        fwd_out + off, DIM*SEQ); off += DIM*SEQ;
                cvt_f16_f32(ac->K,        fwd_out + off, DIM*SEQ); off += DIM*SEQ;
                cvt_f16_f32(ac->V,        fwd_out + off, DIM*SEQ);
#if USE_RMSNORM_FUSION
                cvt_f16_f32(ac->xnorm,    fwd_out + 4*DIM*SEQ, DIM*SEQ);
#endif
                IOSurfaceUnlock(dk.sdpaFwd->ioOut, kIOSurfaceLockReadOnly, NULL);
```

For FFN: skip CPU RMSNorm2, write raw x2, read x2norm from output:

```c
                // CPU: scaled residual
                vDSP_vsma(ac->o_out, 1, &res_alpha, x_cur, 1, ac->x2, 1, (vDSP_Length)(SEQ*DIM));

#if USE_RMSNORM_FUSION
                write_ffn_fused_acts_raw(pls[L].ffnFused_in, ac->x2);
#else
                rmsnorm(ac->x2norm, ac->x2, lw[L].rms_ffn, DIM, SEQ);
                write_ffn_fused_acts(pls[L].ffnFused_in, ac->x2norm, ac->x2);
#endif
                ane_eval_req(dk.ffnFused, plr[L].ffnFused);

                // Read fused output
                IOSurfaceLock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
                _Float16 *ffn_out = (_Float16*)IOSurfaceGetBaseAddress(dk.ffnFused->ioOut);
                off = 0;
                cvt_f16_f32(x_cur,       ffn_out + off, DIM*SEQ);     off += DIM*SEQ;
                cvt_f16_f32(ac->h1,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                cvt_f16_f32(ac->h3,      ffn_out + off, HIDDEN*SEQ);  off += HIDDEN*SEQ;
                cvt_f16_f32(ac->silu_out,ffn_out + off, HIDDEN*SEQ);
#if USE_RMSNORM_FUSION
                cvt_f16_f32(ac->x2norm,  ffn_out + (DIM+3*HIDDEN)*SEQ, DIM*SEQ);
#endif
                IOSurfaceUnlock(dk.ffnFused->ioOut, kIOSurfaceLockReadOnly, NULL);
```

**Step 4: Restage gamma weights after Adam update**

In the dispatch_apply block, after staging forward weights, also restage gamma (since rms_att/rms_ffn weights get updated by Adam):

```c
#if USE_RMSNORM_FUSION
                    stage_sdpa_fwd_gamma(p_pls[L].sdpaFwd_in, p_lw[L].rms_att);
                    stage_ffn_fused_gamma(p_pls[L].ffnFused_in, p_lw[L].rms_ffn);
#endif
```

**Step 5: Update validation forward loop similarly**

The validation loop (line 751-773) does forward-only. Apply the same conditional changes: skip CPU rmsnorm when fused, write raw activations, read xnorm from output (though validation doesn't need xnorm — can skip reading it).

**Step 6: Compile and smoke-test**

Run: `make -C ane clean && make -C ane train_ane && ANE_WALL_TIME=30 python harness_ane.py`
Expected: compiles, runs 30s, no crash, val_loss comparable to baseline

**Step 7: Commit**

```bash
git add ane/stories_mil_dynamic.h ane/stories_io.h ane/train_ane.m
git commit -m "feat: fuse RMSNorm into sdpaFwd and ffnFused ANE kernels"
```

---

### Task 6: INT8 activation quantization — MIL changes

**Files:**
- Modify: `ane/stories_mil_dynamic.h`

**Context:** Add `quantize`/`dequantize` MIL ops between matmul operations inside forward kernels. This reduces L2 SRAM bandwidth for intermediate activations. Only affects forward kernels — backward stays fp16 for gradient precision.

The maderix benchmark uses scale=0x1p-3 (1/8) for quantize/dequantize. We'll use a learned-free fixed scale.

**Step 1: Add INT8 quantize/dequantize to gen_ffn_fused_dynamic**

The best place is around the `gate` tensor (SiLU output × h3) before the W2 matmul. This is the largest intermediate: [HIDDEN, SEQ] = 1536×512 = 786K elements.

After the gate computation and before the W2 matmul reshape:

```c
#if USE_INT8_ACTIVATIONS
    // Quantize gate to int8, then dequantize back — halves SRAM bandwidth for W2 matmul input
    [m appendString:@"        fp16 q_scale = const()[name=string(\"q_scale\"), val=fp16(0x1p-3)];\n"];
    [m appendString:@"        string q_dtype = const()[name=string(\"q_dtype\"), val=string(\"int8\")];\n"];
    [m appendString:@"        fp16 dq_scale = const()[name=string(\"dq_scale\"), val=fp16(0x1p-3)];\n"];
    [m appendFormat:@"        tensor<int8, [1,%d,1,%d]> gate_q = quantize(input=gate,output_dtype=q_dtype,scale=q_scale)[name=string(\"gq\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate_dq = dequantize(input=gate_q,scale=dq_scale)[name=string(\"gdq\")];\n", HIDDEN, SEQ];
    // Use gate_dq instead of gate for W2 matmul
    #define GATE_VAR @"gate_dq"
#else
    #define GATE_VAR @"gate"
#endif
```

Then in the W2 matmul section, use GATE_VAR:

```c
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> g2 = reshape(shape=rg,x=%@)[name=string(\"g2\")];\n", HIDDEN, SEQ, GATE_VAR];
```

(Undefine GATE_VAR after use to avoid macro pollution.)

**Step 2: Add INT8 to gen_sdpa_fwd_dynamic**

Quantize the QKV results before attention (the [DIM, SEQ] tensors before head reshape):

```c
#if USE_INT8_ACTIVATIONS
    [m appendString:@"        fp16 q_scale = const()[name=string(\"q_scale\"), val=fp16(0x1p-3)];\n"];
    [m appendString:@"        string q_dtype = const()[name=string(\"q_dtype\"), val=string(\"int8\")];\n"];
    [m appendString:@"        fp16 dq_scale = const()[name=string(\"dq_scale\"), val=fp16(0x1p-3)];\n"];
    // Quantize Q, K, V to int8 then back — reduces SRAM for head reshape + attention
    [m appendFormat:@"        tensor<int8, [1,%d,1,%d]> qfq = quantize(input=qf,output_dtype=q_dtype,scale=q_scale)[name=string(\"qqf\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf2 = dequantize(input=qfq,scale=dq_scale)[name=string(\"dqf\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int8, [1,%d,1,%d]> kfq = quantize(input=kf,output_dtype=q_dtype,scale=q_scale)[name=string(\"qkf\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf2 = dequantize(input=kfq,scale=dq_scale)[name=string(\"dkf\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int8, [1,%d,1,%d]> vfq = quantize(input=vf,output_dtype=q_dtype,scale=q_scale)[name=string(\"qvf\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf2 = dequantize(input=vfq,scale=dq_scale)[name=string(\"dvf\")];\n", DIM, SEQ];
    // Use quantized versions for attention (but output original qf,kf,vf for backward tap)
    // Reshape quantized QKV to heads
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=hsh,x=qf2)[name=string(\"rq\")];\n", HEADS, HD, SEQ];
    // ... (use qf2, kf2, vf2 for attention computation)
    // ... (but output concat still uses original qf, kf, vf for backward gradient accuracy)
#else
    // ... existing head reshape using qf, kf, vf ...
#endif
```

**Important:** The output concat should still output the ORIGINAL (non-quantized) qf, kf, vf for backward pass gradient accuracy. Only the attention COMPUTATION uses quantized versions.

**Step 3: Compile and smoke-test**

With USE_INT8_ACTIVATIONS=0 (default): verify no change in behavior.
Toggle to 1, recompile: verify it runs without ANE compile error.

Run: `make -C ane clean && make -C ane train_ane && ANE_WALL_TIME=30 python harness_ane.py`

**Step 4: Commit**

```bash
git add ane/stories_mil_dynamic.h
git commit -m "feat: INT8 activation quantization in forward MIL kernels"
```

---

### Task 7: Documentation updates

**Files:**
- Modify: `program_ane.md`
- Modify: `ane/experiment_config.h` (verify comments are clear)

**Step 1: Add feature toggles section to program_ane.md**

Add after the existing optimizer parameters section:

```markdown
**ANE pipeline features** (toggle to experiment — changing these requires recompile but does NOT reset checkpoint):

| Parameter | Default | Notes |
|---|---|---|
| `USE_CHANNEL_FIRST_CPU` | 1 | Stage Wq/Wk/Wv/Wo without CPU transpose. Saves ~75MB RAM, eliminates 32 transposes per restage. No accuracy impact. |
| `USE_RMSNORM_FUSION` | 1 | Compute RMSNorm inside ANE kernels. Eliminates 16 CPU round-trips per step. Numerically equivalent in fp16. |
| `USE_INT8_ACTIVATIONS` | 0 | Quantize activations to int8 between ops in forward kernels. Reduces ANE SRAM bandwidth ~2x. May affect convergence — experiment! |
```

**Step 2: Commit**

```bash
git add program_ane.md
git commit -m "docs: document ANE pipeline feature toggles in program_ane.md"
```

---

## Verification Checklist

After all tasks:

1. `USE_CHANNEL_FIRST_CPU=1, USE_RMSNORM_FUSION=1, USE_INT8_ACTIVATIONS=0`: Compile + 30s run → no crash, reasonable loss
2. `USE_CHANNEL_FIRST_CPU=1, USE_RMSNORM_FUSION=1, USE_INT8_ACTIVATIONS=1`: Compile + 30s run → no crash (loss may differ)
3. `USE_CHANNEL_FIRST_CPU=0, USE_RMSNORM_FUSION=0, USE_INT8_ACTIVATIONS=0`: Compile + 30s run → baseline behavior unchanged
4. Compare ane_util_pct across configurations
5. Full 5-min run with best config, compare val_loss to baseline 3.102
