# Experiment Log — LoViF 2026 AIGC Image Compression (ECCV 2026)

Team: ZeroR · Codabench comp 17125 · Constraint: avg ≤ 0.025 bpp on 100 val images.
Leaderboard score = `PSNR + 10·MS-SSIM − 40·LPIPS(alex) − 40·DISTS` (our local "score" = board + 80).
All compute on Modal (profile `staratnyte0`, volume `aigc-ic`, A10G GPUs).

## Verified facts
- Organizers' eval: RGB PSNR, LPIPS-**alex**, DISTS, MS-SSIM — our local eval matches the board digit-for-digit.
- Board bpp = total bits / total pixels (weighted), slightly looser than our per-image mean → ~0.0007 bpp free headroom.
- Base codec: AEIC (CVPR 2026, Zhang et al., MIT) — SD-Turbo one-step diffusion decoder, ME variant, ckpts ft{2,4,8,16,32}.

## Experiments

| # | Date | What | Result (local score @ bpp) | Verdict |
|---|------|------|-----------------------------|---------|
| E1 | 07-05 | AEIC-ME ft4 zero-shot, val | 109.21 @ 0.0239 | baseline |
| E2 | 07-05 | ft8 / ft2 zero-shot | 106.19 @ 0.017 / 111.28 @ 0.031(over) | rate ladder |
| E3 | 07-05 | Knapsack mix ft2/4/8 → **submission_v1** | **109.55 @ 0.0250** | board 29.55-e |
| E4 | 07-05 | Synth corpus: GenEval+DPG+CVTG-2K+LongText prompts (3938) × {SDXL, PixArt-Σ, Sana} | ~11.5k imgs | FLUX blocked (401, needs HF token) |
| E5 | 07-05 | Fine-tune λ4 (init ft4, 8k steps, +LPIPS-alex loss λ=1, DINO GAN disc, train800+synth) | ckpt5000: 110.70 @ 0.0293 | above-curve per-image, on-curve mean; Modal preempted @~6k, rerun stopped |
| E6 | 07-05 | Fine-tune λ8 (init ft8, 5k steps) | ckpt5000: 108.34 @ 0.0201; ckpt2000: 108.11 @ 0.0198 | +0.6 above baseline curve at low rate |
| E7 | 07-05 | 7-way knapsack → **submission_v2** | **109.89 @ 0.02497** | board: — |
| E8 | 07-05 | Gradient latent TTO through frozen decoder (STE, crop-based, 4 variants: fp32/bf16, lr/hinge sweeps) | diverges: bpp ↑, LPIPS ↑ | **ABANDONED** — pathological STE grads through 4-group autoregressive conditioning; code in AEIC/src/refine.py |
| E9 | 07-05 | +4 intermediate ckpts (aigc4_3000/4000, aigc8_3000/4000), 11-way Lagrangian+slack-fill → **submission_v3** | **109.94 @ 0.02500** | board 29.9384, rank 2/8 |

## Leaderboard snapshot 2026-07-05
1. UnoChen 32.31 (PSNR 27.21, LPIPS 0.065, DISTS 0.036 @ 0.0222 bpp, 2.2s) — gap 2.37
2. **ZeroR 29.94** (ours)
3. yingliang 25.56

## Gap analysis vs UnoChen
LPIPS −0.031 (1.25 pts) + DISTS −0.014 (0.58) + PSNR +0.48 + MS-SSIM +0.006, at lower bpp. Better model, not better packing. → round-2 training.

| E10 | 07-05 | Weighted-bpp knapsack (board bpp = Σbits/Σpx, verified vs board's 0.0243) → **submission_v4** | local 110.53 @ w-bpp 0.0249 | **board 30.5317 — predicted 30.5317 exactly. Local eval = perfect oracle.** rank 2, gap 1.78 |

| E11 | 07-06 | Round-2 λ4 done (18k steps incl. preempt-restart overshoot): r2_18000 112.18 @ 0.037, r2_12000 112.11 @ 0.0369 | strong high-rate points | keep |
| E14 | 07-06 | λ2 fine-tune done: r2b_7000 112.77 @ 0.0456 | premium point for small imgs | keep |
| E15 | 07-06 | 16-run weighted remix → **submission_v5**: local 110.82, board est **30.82** @ w-bpp 0.0249 | r2 ckpts carry 42/100 imgs | **board 30.8172 confirmed — third exact prediction.** rank 2, gap 1.49 |

| E13 | 07-08 | Round-2 λ8 on sd35-augmented corpus (8k steps): r3l8_8000 110.15 @ 0.0259 | +0.35 above curve | keep |
| E18 | 07-09 | Latent TTO FIXED — frozen-conditioning gradients (means/scales detached from init pass): stable bpp, +0.32/img mean; all 100 val refined at v5-chosen ckpts | refined set 110.91 @ 0.0311 | works |
| E19 | 07-09 | Grand remix (16 base + r3l8 + refined) → **submission_v6**: local 111.07, board est **31.07** @ w-bpp 0.0249; refined carries 66/100 | pending board |

## Queued / running
- E12: FLUX dead on A10G (12B transformer > 22GB even offloaded) — replaced with SD3.5-medium; corpus complete (3938 sd35 + 3938 sdxl + pixart + sana).
- E13: round-2 λ8 (init aigc8_5000, sd35-augmented corpus) — next GPU run.
- E16: round-3 λ4 on sd35+sdxl-augmented corpus.
- E17: decoder-side enhancer (train pairs: r2_18000_train + aigc8_5000_train recon of train-800) — pairs ready, trainer TBD.
- E20: TTO round on refined-selection loop (refine against v6 plan incl. r3l8 picks).

## Infra gotchas (hard-won)
- compressai 1.2.8 renamed EntropyBottleneck buffers (`_matrix0`→`matrices.0`); bidirectional remap patched into both loaders.
- torch.compile (inductor, torch 2.1.2) crashes on tiled VAE — compile_model=False everywhere.
- vision_aided_loss 'dinov3' needs torch≥2.4 → used 'dino'; hub download races with 4 DDP procs → single-proc warmup.
- Modal: A100/L40S gated (no payment method), A10G ok; long jobs get preempted and silently restart from scratch — checkpoints every 1000 steps are the recovery story.
- finetune.py appends λ to exp_name (`AEIC_ME_aigc8` + λ8 → `AEIC_ME_aigc88_*.pkl`).
- gdown CLI on Modal images is stale — use python API `gdown.download(id=...)`.
