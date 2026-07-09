# Experiment Log — LoViF 2026 AIGC Image Compression (ECCV 2026)

Team: ZeroR · Codabench comp 17125 · Constraint: avg ≤ 0.025 bpp on 100 val images (board bpp = Σbits/Σpixels).
Board score = `PSNR + 10·MS-SSIM − 40·LPIPS(alex) − 40·DISTS` (our local "score" = board + 80).
All compute on Modal (profile `staratnyte0`, volume `aigc-ic`, A10G GPUs). Repo: StarAtNyte/lovif-aigc-compression.

## Verified facts
- Organizers' eval matches ours digit-for-digit: RGB PSNR, LPIPS-**alex**, DISTS, MS-SSIM. Local eval = exact leaderboard oracle (4/4 exact predictions).
- Board bpp is **weighted** (total bits / total pixels) — looser than per-image mean; small images can take expensive rate points.
- Base codec: AEIC (CVPR 2026, Zhang et al., MIT) — SD-Turbo one-step diffusion decoder, ME variant, zero-shot ckpts ft{2,4,8,16,32}.

## Experiments

| # | Date | What | Result (local score @ bpp) | Verdict / board |
|---|------|------|-----------------------------|-----------------|
| E1 | 07-05 | AEIC-ME ft4 zero-shot on val | 109.21 @ 0.0239 | baseline |
| E2 | 07-05 | ft8 / ft2 zero-shot | 106.19 @ 0.017 / 111.28 @ 0.031 | rate ladder |
| E3 | 07-05 | Knapsack mix ft2/4/8 → **v1** | 109.55 @ 0.0250 (mean-bpp) | board 29.94 |
| E4 | 07-05 | Synth corpus: GenEval+DPG+CVTG-2K+LongText prompts (3,938) × {SDXL, PixArt-Σ, Sana} | ~11.5k imgs | FLUX 401 (token needed) |
| E5 | 07-05 | Fine-tune λ4 "aigc4" (init ft4, 8k steps, +LPIPS loss, DINO GAN, train800+synth) | ckpt5000: 110.70 @ 0.0293 | Modal preempted @~6k; ckpt5000 kept |
| E6 | 07-05 | Fine-tune λ8 "aigc8" (init ft8, 5k steps) | ckpt5000: 108.34 @ 0.0201 | +0.6 above curve |
| E7 | 07-05 | 7-way knapsack → **v2** | 109.89 @ 0.02497 | not submitted (superseded) |
| E8 | 07-05 | Gradient latent TTO, naive (STE through 4-group autoregressive conditioning; 4 variants) | diverges: bpp↑, LPIPS↑ | abandoned → fixed later in E18 |
| E9 | 07-05 | +4 intermediate ckpts, 11-way Lagrangian → **v3** | 109.94 @ 0.02500 | board **29.9384**, rank 2 |
| E10 | 07-05 | Weighted-bpp knapsack (board's bpp definition, verified vs board 0.0243) → **v4** | 110.53 @ w-bpp 0.0249 | board **30.5317** — exact prediction #2 |
| E11 | 07-06 | Round-2 λ4 "r2" (init aigc4_5000, 16k+ steps, real-800 ×8 oversample, λ_lpips 2.0, λ_dists 1.5) | r2_18000: 112.18 @ 0.037; r2_12000: 112.11 @ 0.0369 | strong high-rate points |
| E12 | 07-06→08 | FLUX dead on A10G (12B > 22GB even offloaded) → SD3.5-medium corpus instead | +3,938 sd35 imgs (sdxl also completed: 3,938) | corpus total ~15.7k |
| E13 | 07-08 | Round-2 λ8 "r3l8" (init aigc8_5000, sd35-augmented corpus, 8k steps) | r3l8_8000: 110.15 @ 0.0259 | +0.35 above curve |
| E14 | 07-06 | λ2 fine-tune "r2b" (init ft2, 6k+ steps) | r2b_7000: 112.77 @ 0.0456 | premium point for small imgs |
| E15 | 07-06 | 16-run weighted remix → **v5** | 110.82 @ w-bpp 0.0249 | board **30.8172** — exact prediction #3 |
| E18 | 07-09 | Latent TTO **fixed**: freeze conditioning (means/scales detached from init pass), gradients only through decode path | stable bpp; +0.32/img mean; all 100 val refined at v5-chosen ckpts → refined set 110.91 @ 0.0311 | works; +0.49 on winning images |
| E19 | 07-09 | Grand remix (16 base + r3l8 + refined; refined carries 66/100) → **v6** | 111.07 @ w-bpp 0.0249 | board **31.0684** — exact prediction #4. rank 2, gap to UnoChen 1.24 |
| E21 | 07-09 | Tiling-seam ablation on r2_18000 (vae_tile 160→224, latent_tile 96→128, overlap 32→48) | 112.1792 @ 0.037 vs baseline 112.18 @ 0.037 — **no effect** | ruled out, zero-cost test |
| E16 | 07-09 | Round-3 λ4 "r3l4" (init r2_18000, sd35+sdxl-augmented corpus, 8k steps) done | r3l4_8000: 112.22 @ 0.0373 | marginal (+0.04 vs r2_18000) |
| E22 | 07-09 | 18-run remix incl. r3l4_8000 → **v7** | 111.0739 @ w-bpp 0.0249 | board est **31.0739** (+0.006 over v6, near-noise) |

## Leaderboard 2026-07-09 (honest entries)
1. UnoChen 32.3084 (PSNR 27.21, LPIPS 0.065, DISTS 0.036 @ 0.0222, 2.2s)
2. **ZeroR 31.0684** (PSNR ~27.1, LPIPS ~0.085, DISTS ~0.046 @ 0.0249)
3. yingliang 26.91
- "anish 40.83" = fake (jumped from −5.45 overnight; PSNR 33 + DISTS 0.0155 @ 0.021 bpp is beyond SOTA in both axes simultaneously; val GT is public and dev server never verifies bitstream→reconstruction). Dies in final phase (decoder verified, human eval).

| E23 | 07-09 | OCR text diagnostic (EasyOCR boxes on val GT, crop-metric vs whole-image on v7 recon) | 52/100 imgs have text. **LPIPS text −0.022 (better!), DISTS text +0.0717 (much worse)** | DISTS is the real text weak point, not LPIPS — targeted fix |
| E24 | 07-09 | TTO extended: OCR-biased crop sampling (70% of iters center on a text box) + 3x DISTS weight on text-region iters | queued | targets the E23 finding directly |

## Queued / running
- E16: round-3 λ4 "r3l4" (init r2_18000, sd35+sdxl-augmented corpus, 8k steps) — launched 07-09.
- E20: TTO pass for v6's r3l8_8000 picks (18 imgs) → separate out_dir `refined2` (skip-guard collision with v5-refined dir). Relaunch pending.
- E17: decoder-side enhancer (legal 5GB-decoder lever): train pairs = {r2_18000, aigc8_5000} reconstructions of train-800 → GT; restoration net with 40·LPIPS+40·DISTS loss; ships inside decoder. Pairs generated (`runs/train/*_train`), trainer TBD.
- Test-phase prep (deadline 07-18): decoder package ≤5GB (SD-Turbo fp16 ~2.5GB + ckpts), README, factsheet.

## Gap to UnoChen (1.24)
LPIPS −0.020 (≈0.8 pts) + DISTS −0.010 (≈0.4 pts); PSNR/MS-SSIM now at parity. E17 enhancer targets exactly these two.

## Infra gotchas (hard-won)
- compressai 1.2.8 renamed EntropyBottleneck buffers (`_matrix0`→`matrices.0`); bidirectional remap patched into both loaders.
- torch.compile (inductor, torch 2.1.2) crashes on tiled VAE — compile_model=False everywhere.
- vision_aided_loss 'dinov3' needs torch≥2.4 → used 'dino'; hub download races with 4 DDP procs → single-proc warmup.
- Modal: A100/L40S gated (no payment method), A10G ok; long jobs get preempted and restart the function from scratch — per-minute `vol.commit()` watcher + per-image skip-guard in refine make preemptions cheap; ckpts every 1000 steps for training.
- finetune.py appends λ to exp_name (`AEIC_r2b_` + λ2 → `AEIC_r2b_2_*.pkl`).
- gdown CLI on Modal images is stale — use python API `gdown.download(id=...)`.
- TTO gradient rule: never backprop through the 4-group autoregressive conditioning — freeze means/scales from the init pass (E8 diverged; E18 works).
- Naive-vs-weighted bpp: always check the organizer's exact arithmetic against a known submission before optimizing the constraint.
