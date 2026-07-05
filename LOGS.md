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

## Queued / running
- E10: weighted-bpp knapsack (board's bpp definition) → resubmit, free ~+0.1–0.2.
- E11: round-2 fine-tune λ4: init aigc4_5000, 16k steps, real-800 oversampled 8× (real data = actual 10 challenge generators), λ_lpips 2.0, λ_dists 1.5.
- E12: FLUX.1-schnell corpus (HF token via Modal secret) → round-3 data.
- E13: round-2 λ8 after E11.

## Infra gotchas (hard-won)
- compressai 1.2.8 renamed EntropyBottleneck buffers (`_matrix0`→`matrices.0`); bidirectional remap patched into both loaders.
- torch.compile (inductor, torch 2.1.2) crashes on tiled VAE — compile_model=False everywhere.
- vision_aided_loss 'dinov3' needs torch≥2.4 → used 'dino'; hub download races with 4 DDP procs → single-proc warmup.
- Modal: A100/L40S gated (no payment method), A10G ok; long jobs get preempted and silently restart from scratch — checkpoints every 1000 steps are the recovery story.
- finetune.py appends λ to exp_name (`AEIC_ME_aigc8` + λ8 → `AEIC_ME_aigc88_*.pkl`).
- gdown CLI on Modal images is stale — use python API `gdown.download(id=...)`.
