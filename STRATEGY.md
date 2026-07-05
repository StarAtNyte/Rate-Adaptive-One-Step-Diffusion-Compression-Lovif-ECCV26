# LoViF 2026 AIGC Image Compression — Winning Strategy

Challenge: compress 100 AIGC images (1K/2K PNG) at **avg ≤ 0.025 bpp**.
Score = `PSNR + 10·MS-SSIM + 40·(1−LPIPS) + 40·(1−DISTS)`. Dev deadline 2026-07-18 (decoder ≤5GB), test by 07-23.

## 1. Score sensitivity (drives everything)

| Metric | Δ needed for +1 pt | Realistic spread between methods @0.025bpp | Points at stake |
|---|---|---|---|
| PSNR | +1 dB | ~3–5 dB | ~4 |
| MS-SSIM | +0.1 | ~0.05 | ~0.5 (nearly irrelevant) |
| LPIPS | −0.025 | ~0.15–0.25 (VVC vs diffusion) | ~6–10 |
| DISTS | −0.025 | ~0.08–0.15 | ~3–6 |

→ LPIPS+DISTS dominate; PSNR secondary; MS-SSIM noise. **Generative one-step-diffusion decoder with a fidelity branch is the right family.** Pure VVC/MSE codecs lose badly; pure text-to-image (PerCo-style) loses PSNR. Human eval in final phase also favors generative + text-readability.

## 2. Prior art landscape (as of Jul 2026)

- **AEIC** (CVPR 2026, github.com/LuizScarlet/AEIC, MIT, arXiv:2512.12229) — SOTA in exactly 0.005–0.035 bpp. SD-Turbo one-step decoder, dual-branch (fidelity + perceptual), trained w/ **overlap-chunked edge-aware DISTS loss + DINOv3-based GAN discriminator**. Checkpoints released. Trains on 4×3090. Same authors as StableCodec (ICCV 2025, github.com/LuizScarlet/StableCodec).
- **DiffO** (arXiv:2506.16572, github.com/Freemasti/DiffO) — one-step, best PSNR-among-perceptual: @0.0294bpp PSNR 24.41, LPIPS 0.319, DISTS 0.133.
- **OSCAR** (arXiv:2505.16091) — one-step multi-rate; strong LPIPS/DISTS.
- **GLC** (arXiv:2505.16177) — VQ-latent transform coding, <0.04bpp.
- **SPRDiff** (arXiv:2606.01608) — triple-encoder diffusion, <0.03bpp, code promised.
- **ResULIC** (arXiv:2505.08281) — semantic residual + compression-aware diffusion.
- **TextBoost** (arXiv:2603.04115) — OCR text as side-info + attention-guided fusion at ultra-low bpp. Directly relevant: 44% of challenge data is text-rendering prompts (CVTG-2K 280 + LongText-Bench 160).
- **Dual-latent collaborative decoding** (arXiv:2605.14391) — fidelity/perception interpolation knob.
- CLIC-winner classic tricks: per-image latent refinement (Campos CVPRW'19), content-adaptive optimization, importance-map bit allocation.

## 3. The winning technique (composite)

**Base: AEIC-ME fine-tuned for this exact score on AIGC data.** Layers of edge, ranked by expected points:

1. **Domain fine-tune on synthetic AIGC corpus (big edge, ~free).** 800 train images is tiny — but the organizers disclosed their recipe: prompts from GenEval/DPG-Bench/CVTG-2K/LongText-Bench through 10 T2I models. Regenerate a 20–50k-image clone corpus with open models (FLUX.1-dev/schnell, SD3.5, SDXL, PixArt, Sana, HiDream…) on Modal. Extra data allowed with factsheet disclosure. Nobody-else-bothers advantage.
2. **Direct score-loss fine-tune.** Replace generic loss with challenge score surrogate: `α·MSE + β·(1−MS-SSIM) + 40·LPIPS + 40·DISTS(edge-aware)` + GAN. Optimizing the exact eval metric ~always wins leaderboards.
3. **Encoder-side per-image latent refinement (test-time opt).** Only the *decoder* is frozen/submitted; encoder compute is unlimited. For each val/test image, gradient-descend the quantized latents (STE) through frozen decoder to maximize the score directly. Worth several points; classic CLIC winner move; fully legal.
4. **Dataset-level bit allocation (knapsack).** Constraint is *average* bpp over the dataset, not per image. Encode every image at ~4–6 rate points, then pick per-image rates maximizing Σscore s.t. Σbits ≤ budget (greedy on marginal score/bit). Easy +1–3 pts vs uniform rate.
5. **Text-region handling.** OCR (PaddleOCR) → importance map boosting bits/loss weight on text; optionally TextBoost-style OCR-string side-info fused in decoder. Helps DISTS/LPIPS on 44% of images and is decisive for the human-eval perceptual awards.
6. **Fidelity/perception knob per image.** Dual-branch weight tuned per image (graphics/flat images → fidelity branch ↑ PSNR; texture-rich → perceptual).

Fallback if AEIC fine-tune misbehaves: DiffO or StableCodec base; same wrapper tricks (3)(4)(5) apply to any base.

## 4. Execution plan (Modal)

- Vol `aigc-ic`: train (800), val (100) from Google Drive; checkpoints; synthetic corpus.
- Phase A (day 1–2): env + AEIC checkpoints on Modal; zero-shot baseline on val at 0.025bpp; submit → leaderboard calibration.
- Phase B (day 2–5): synthetic AIGC corpus generation (FLUX-schnell + SD3.5 + SDXL etc., prompts from the 4 public benchmark prompt sets).
- Phase C (day 4–9): score-loss fine-tune (A100/H100), eval loop on val with exact metrics (pyiqa: lpips-vgg? — confirm which LPIPS net organizers use; default lpips-alex vs vgg matters!).
- Phase D (day 8–12): latent refinement + knapsack allocation + text weighting; ablate each on val leaderboard.
- Phase E (by 07-18): package decoder (<5GB: SD-Turbo UNet+VAE ~2.5GB fp16 + codec ~0.5GB — fits), README, submit code; 07-20+ run test set, factsheet.

Open items: confirm organizers' LPIPS backbone + eval script (challenge "Provided Resources: scripts"); grab submission-example zip; register team on Codabench.

## Sources
- https://arxiv.org/abs/2512.12229 (AEIC) · https://github.com/LuizScarlet/AEIC
- https://arxiv.org/abs/2506.21977 (StableCodec) · https://github.com/LuizScarlet/StableCodec
- https://arxiv.org/html/2506.16572v1 (DiffO) · https://github.com/Freemasti/DiffO
- https://arxiv.org/pdf/2505.16091 (OSCAR) · https://arxiv.org/abs/2505.16177 (GLC)
- https://arxiv.org/pdf/2606.01608 (SPRDiff) · https://arxiv.org/pdf/2505.08281 (ResULIC)
- https://arxiv.org/pdf/2603.04115 (TextBoost) · https://arxiv.org/pdf/2605.14391 (dual-latent)
- https://openaccess.thecvf.com/content_CVPRW_2019/papers/CLIC%202019/Campos_Content_Adaptive_Optimization_for_Neural_Image_Compression_CVPRW_2019_paper.pdf
- https://arxiv.org/html/2410.09834v3 (AIGIF, organizers' group)
