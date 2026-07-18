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
| E24 | 07-09 | TTO with OCR-biased crop sampling (70% iters center on text box) + 3x DISTS weight on text iters, 52 affected images, 100 iters | huge per-image gain: mean Δscore +1.61 vs plain TTO, 44/52 improved (up to +6.76 on one image) — but bpp also rose ~0.005-0.03 on the big winners | works, but costly in bits |
| E25 | 07-09 | 18-way remix incl. refined_text → **v8** | local 111.096 @ w-bpp 0.0249 | board est **31.0962** (+0.02 over v7) — real but small; existing r3l8/r3l4/refined candidates already captured most of the raw gain under budget, only 9/52 text images win the global slot |

| E26 | 07-09/10 | E17 decoder-side enhancer: 1814 pairs, 8 RRDB residual CNN (~1.5M params), L1+LPIPS+DISTS loss, 8k steps to convergence | +0.22 avg score gain, UNIFORM across every candidate type (0.19-0.25 range), zero bpp cost (post-decode) | ships in final decoder, stacks with everything |
| E27 | 07-10 | Grand remix with all enhanced candidates → **v9** | local 111.286 @ w-bpp 0.0249 | board est **31.2863** (+0.19 over v8) |

| E28 | 07-10 | Enhancer round 2: EA-DISTS + DINO GAN, resumed from round-1 8k ckpt, full 2400 pairs, 10k steps (batch3/patch256 for VRAM) | GAN loss **regresses** vs round-1 alone: best round-2 ckpt (1000) = +0.16 vs round-1's ck8000 = +0.22. Confirms literature warning (NTIRE 2026: adversarial training unstable under multi-metric optimization). GAN not adopted. | negative result, real finding |
| E29 | 07-10 | Isolated EA-DISTS-only retest (gan_w=0), resumed from round-1 8k, 3k steps | ck1000 +0.2108, ck3000 +0.2046 — plateaus, does NOT beat round-1 plain (+0.2204). **Neither GAN nor EA-DISTS improved on plain L1+LPIPS+DISTS.** | negative result — proven recipe stays: enhancer_out/enhancer_8000.pt (used in v9) |
| E30 | 07-10 | Continue proven plain recipe (no GAN, no EA) another 10k steps from ck8000, lr 1e-4, full 18k total | ckpt10000: **+0.2678** vs base (round-1's ck8000 was +0.2204) — real gain, confirms it's TRAINING DURATION not loss-function additions that helps | new champion enhancer |
| E31 | 07-10 | Applied champion enhancer (18k steps) to all 8 v9-mix candidates | uniform +0.22 to +0.31 (avg ~+0.27), beats round-1's uniform +0.22 | keep |
| E32 | 07-10 | Final grand remix with champion enhancer → **v10** | local 111.3249 @ w-bpp 0.0249 | board est **31.3249** (+0.04 over v9) |

| E33 | 07-11 | Enhancer continued (round 5, 15k more steps, total 23k from base) | +0.2574 vs v4's +0.2678 at 18k — **plateaued/regressed slightly**. More steps beyond 18k don't help; small 2400-sample dataset limits capacity. | v4 ck10000 (18k) remains champion, no change |
| E34 | 07-11 | Text-TTO with 4x tighter rate penalty (rate_w 20000→80000) to get DISTS gain within-budget instead of over-budget | bpp barely moved (0.0436→0.0435) but score dropped (111.38→111.30) — rate_w wasn't the actual constraint on latent growth; tightening it just hurt reconstruction quality for no rate benefit | negative, reverted to rate_w=20000 (already in v10) |

## CRITICAL: decoder size budget (discovered 07-11)
v10's mix spans ~15 distinct base checkpoints (TTO seeded per-image from whichever knapsack round picked it). This CANNOT ship: SD-Turbo full download is 13.3GB; only unet+vae subfolders are used by AEIC (confirmed via base_model.py), and only the **fp16** safetensors are needed (fp32 duplicates + unused text_encoder/tokenizer wasted). Fixed infra = unet fp16 (1.73GB) + vae fp16 (0.17GB) + AdcSR halfDecoder (0.38GB) = **2.28GB**. Each AEIC-ME checkpoint = 616MB → budget allows **exactly 4 checkpoints** (2.72GB / 0.616GB = 4.4).
Chosen 4-checkpoint set (best per rate tier): r2b_7000 (0.0456), r3l4_8000 (0.0373), r3l8_8000 (0.0259), aigc8_5000 (0.0201). Total package: 4×616MB + 2.28GB = **4.63GB, fits with ~260MB spare**.
Raw 4-ckpt knapsack: 30.84 board est (vs v10's dev-only 31.32 using unshippable 15-ckpt mix). Re-running full TTO (100 imgs, not just 52) + enhancer stacking on this restricted, shippable set — E35.

| E35 | 07-11 | Full TTO (100 imgs) seeded only from the 4 shippable checkpoints + champion enhancer applied to both TTO'd and raw outputs; final knapsack over these 6 candidates only | local 111.3127 @ w-bpp 0.0249 | board est **31.3127** — only −0.012 vs v10's unshippable 15-ckpt mix. TTO+enhancer carry nearly all the value; checkpoint diversity was mostly redundant. **submission_v11_shippable.zip** |
| E36 | 07-11 | Actual decoder package assembled (fp16 unet+vae, halfDecoder, 4 ckpts, enhancer, src) and measured | **4.756 GB real, confirmed under 5GB limit** (244MB spare) | packaging validated, not just estimated |
| E37 | 07-11 | Exhaustive search over all C(15,4) 4-checkpoint combos (free, no GPU) | best: {aigc8_5000, r2_18000, r3l4_8000, r3l8_8000} = 110.86 raw vs old set's 110.84 | swap r2b_7000→r2_18000 |
| E38 | 07-11 | Generated r3l4_8000 train-800 pairs (missing from enhancer's training distribution); retrained enhancer from scratch on all 4 matched checkpoints (3200 pairs, 18k steps) — running; meanwhile full TTO (100 imgs) reseeded from optimal E37 combo | TTO alone: 110.999 @ 0.0306 (was 110.939) | +0.06 from better seeds alone |
| E39 | 07-11 | Interim remix: optimal-seed TTO + OLD enhancer (matched enhancer still training) → **v12** | local 111.339 @ w-bpp 0.0249 | board est **31.3391** — already beats v10's unshippable 31.3249, on just 4 checkpoints. Gap to UnoChen: 0.97 |
| E40 | 07-11 | Enhancer retrained from scratch on matched 4-ckpt distribution (3200 pairs incl. new r3l4_8000, 18k steps), applied to TTO output + all 4 raw ckpts | modest but real gain over old (mismatched) enhancer: +0.013 on TTO candidate | distribution-match confirmed, small effect |
| E41 | 07-11 | Final remix, everything (TTO + old-enh + matched-enh, raw + enhanced ×2) → **v13, submission_v13_shippable.zip** | local 111.3555 @ w-bpp 0.0249 | board est **31.3555**. Gap to UnoChen: 0.955. Still 4-checkpoint shippable (4.76GB) |
| E42 | 07-11 | Investigated LoRA-delta packaging trick (ship small unet deltas + shared base instead of merged weights) to unlock more checkpoints | checkpoint = 122MB LoRA (already delta-only, not full unet) + 494MB codec (genuinely per-checkpoint, not shareable) — no free capacity to unlock. AEIC-SE 5th-checkpoint idea also dead-ended: no codec_type support in finetune.py, would need new training code | both abandoned, no shortcut exists |
| E43 | 07-11 | Text-biased TTO (rate_w=20000, 150 iters) on the 52 text images, seeded from the correct shippable 4-ckpt set (never tried this exact combo before) + matched enhancer | text-TTO alone 110.10@0.0324; +matched-enh 110.33@0.0324 | wins 23/100 images in final remix |
| E44 | 07-11 | Complete final remix (all TTO variants × both enhancers × 4 raw+enhanced) → **v14, submission_v14_shippable.zip** | local 111.3774 @ w-bpp 0.0249 | **board 31.3774 confirmed exact.** Gap to UnoChen: **0.933** |
| E45 | 07-11 | 16-block (2x capacity) enhancer trained on matched distribution, 18k steps, tested at full convergence | 111.2475 vs 8-block's 111.2689 — **slightly worse**, capacity was never the bottleneck | negative, closes out enhancer architecture search |

| E46 | 07-12 | Knapsack CAP margin sweep (0.0249→0.02499, free/no-GPU) → **v15** | local 111.3981 @ w-bpp 0.02498 | board est **31.3981** (+0.02 free) |
| E47 | 07-12 | TTO at 300 iters (3x depth vs previous 100-150) on all 100 images, seeded from correct 4-ckpt set + matched enhancer | **111.5264 @ 0.0312 — real +0.26 gain**, far bigger than any enhancer tweak. Wins 60/100 in final remix. | reverses "search space exhausted" conclusion — TTO depth was NOT saturated |
| E48 | 07-12 | Final remix incl. 300-iter TTO → **v16, submission_v16_shippable.zip** | local 111.5141 @ w-bpp 0.024979 | board est **31.5141**. Gap to UnoChen: **0.796** |
| E49 | 07-12 | 300-iter text-biased TTO (matching new depth) on 52 text images | running | extend the E47 finding to text-weighted path |

| E50 | 07-12 | 300-iter text-biased TTO (recovering from a mid-run crash on r3l8_8000, resumed via skip-guard) | 110.43@0.0329 vs 150-iter's 110.33@0.0324 | +0.096, consistent with main trend |
| E51 | 07-12 | 500-iter sanity test on 10-image subset before full commit | mean Δ+0.19 vs 300-iter, still positive | real headroom, committed to full run |
| E52 | 07-12 | Full 500-iter TTO (100 images, 3 shards) + matched enhancer | 111.5532@0.0319 vs 300-iter's 111.5264@0.0312 | +0.027 raw, but knapsack-level gain tiny — extra iters cost more bits than budget can afford for most images |
| E53 | 07-12 | Final remix incl. 500-iter candidates → **v17, submission_v17_shippable.zip** | local 111.5270 @ w-bpp 0.024979 | board est **31.5270** (+0.013 over v16) — 300-iter still wins 44/100 vs 500-iter's 19. **TTO-depth curve has clearly bent: 150→300 gave +0.26, 300→500 gives +0.013.** |

## Conclusion (07-12): TTO depth diminishing returns confirmed at ~300 iters
150→300 iters: +0.26 (large). 300→500 iters: +0.013 (near-zero after budget competition). Further iteration increases (1000+) not worth pursuing — cost in bits exceeds knapsack-level value for the vast majority of images. 300 iters is the practical sweet spot for this setup.

## Conclusion (07-11, REVISED 07-12): TTO depth was the real remaining lever
The E33-E45 "exhausted" conclusion was premature — those experiments varied loss functions and enhancer capacity, but never simply ran TTO longer. 300 iters (vs 100-150) gave +0.26, the single biggest gain since the shippable-set rebuild. Still open: does 500+ iters help further, or does this plateau too?
Tried and ruled out: GAN loss (E28), EA-DISTS (E29), more steps beyond 18k (E33), tighter rate penalty (E34), bigger capacity (E45). Only real wins: training duration to 18k (E30), distribution-matched retraining (E40), correct-seed text-TTO (E43). **submission_v14_shippable.zip (board est 31.3774) is the practical ceiling for this architecture** without a genuinely different codec family (OneDC, blocked on manual download) or a new base-training round. Gap to UnoChen: 0.933.

## Research notes (OneDC evaluation, not pursued)
OneDC (NeurIPS 2025, github.com/onedc-codec/onedc) considered as a genuinely-diverse second codec family for the knapsack (unlike StableCodec, which is AEIC's own predecessor by the same authors — correlated failure modes, low ensemble value). Blocked by: checkpoints behind OneDrive link (hard to script-download headless), requires torch 2.5.0 (conflicts with our pinned 2.1.2, needs a fresh Modal image), and bitstream realism (real entropy-coded vs simulated bpp) undocumented. Multi-hour integration with uncertain payoff — flagged for explicit user go-ahead rather than silently spending the session on it.

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

| E54 | 07-12 | crop_ly TTO experiment: expose --crop_ly (default 16=512px crop) to test bigger context per iteration | 8-img subset @300iters: crop24 vs crop16 → **+0.12/img** (dists 0.0430 vs 0.0476, notably better; bpp +2.4%) | positive signal, launched full 100-img run |

| E55 | 07-12 | Full 100-img crop_ly=24 TTO (300 iters) + matched enhancer, evaluated | 111.5709 raw mean @ 0.03192 naive bpp (vs crop16-300's 111.2242@0.03115) | positive, real per-candidate gain |
| E56 | 07-12 | Rebuilt knapsack solver using TRUE weighted bpp (bits/pixels via real image dims, not mean of per-image bpp — caught a solver bug: naive mean-bpp constraint badly misallocates when val images have very different resolutions) | Added crop24 candidate to the 7-candidate mix (300crop16, 500crop16, 4 raw+enh, text_long) | **v18**: local 111.5207 @ w-bpp 0.024947, crop24 wins 25/100 images. submission_v18_crop24mix.zip |

| E57 | 07-12 | **v18 board confirmed: 31.5207** — WORSE than v17's 31.5270 (−0.0063) | crop_ly=24 candidate net-negative in the real knapsack despite +raw-score-per-candidate; local solver reproduction of v17's exact candidate set scored 111.4834 vs v17's real 111.5270 (0.04 gap — solver/pipeline mismatch not fully diagnosed), crop24 addition didn't close it | **v17 remains best submission.** crop-window-size lever closed out, negative. |

| E58 | 07-13 | Replaced the historical greedy/Lagrangian remix with an exact multiple-choice MILP over every shippable raw, enhancer, 300/500-iter TTO, text-TTO, and crop24 candidate using true file bits and source dimensions | **111.5645 @ weighted-bpp 0.0249899** (board est **31.5645**), +0.0375 over v17 | new local best; four base checkpoints remain unchanged, extra enhancer weights are small/package-feasible |
| E59 | 07-13 | Matched budget ablation: same four shippable checkpoints + matched enhancer, exact MILP, with vs without TTO | without TTO **31.1755 @ 0.0249898**; with TTO **31.5065 @ 0.0249746** | TTO contributes **+0.3310** at the legal bitrate; high-rate non-TTO 32.4675 result is invalid alone at 0.0370 bpp |
| E60 | 07-13 | 12-image/four-rate-tier TTO ablation, reduce MSE weight 1000→300→0 at 300 iterations | matched scores: **29.8763 → 29.8205 → 29.7903**; bitrate essentially unchanged | negative; MSE stabilizes the latent search and improves the combined metric |
| E61 | 07-13 | Global DISTS surrogate weight calibration 40→80 and 120 (same matched shard; 120 completed 6 images before client interruption) | DISTS-80: **29.7435 vs 29.8763 baseline**, no DISTS improvement; DISTS-120: **29.7808 vs 29.8142** on common 6 | negative; higher crop-level DISTS weight damages LPIPS/PSNR and does not transfer to whole-image DISTS |
| E62 | 07-13 | OSCAR one-step diffusion codec screen on 12 validation images, official checkpoint and SD2.1 base, exact source-size reflect padding/cropping, published rate tiers | board-equivalent: **-11.5602 @ 0.001902**, **2.4358 @ 0.009802**, **16.1365 @ 0.031302** | decisively noncompetitive vs AEIC 31.5645; reject. Released OSCAR inference also discards VQ indices and provides nominal-rate reconstruction rather than a serialized bitstream |

## Conclusion (07-13): AEIC/TTO local objective is saturated
Exact selection recovered a small free gain to 31.5645, but loss-weight calibration failed and the remaining gap to 33.5 is ~1.94. Incremental AEIC enhancer/TTO tuning is not a credible route to the target. Next branch: integrate a genuinely different one-step diffusion codec (OSCAR first because one checkpoint supports multiple rates; OneDC second if its OneDrive weights can be downloaded reproducibly), evaluate real entropy-coded outputs, and add them to the exact weighted knapsack.

| E63 | 07-13 | Re-verified E58 MILP claim with widened candidate pool (9 candidates incl. 150-iter text variant) | best reproducible: **111.5395 @ 0.024990** — still below v17's board-confirmed 111.5270 local... marginal, not a clean win either way | E58's 31.5645 claim not reproducible from available candidate metrics; crop24 is not a live lever under any solver. v17 stays best. |
| E64 | 07-13 | Multi-seed TTO ensemble test (seed=1 vs seed=0, 300 iters crop16, 8-img subset) | seed1 loses 6/8 images, mean **−0.22/img**; wins are marginal (+0.04, +0.28), one big loss (−1.30) | negative — TTO convergence is seed-insensitive, not a useful diversity lever. Closed. |

## Conclusion (07-13, cont.): search space re-confirmed exhausted
Crop window (E54-57), loss-weight calibration (E60-61), and TTO seed diversity (E64) all closed negative/marginal. OSCAR rejected (E62). v17 (board 31.5270) remains the practical ceiling for this architecture. Remaining real lever is OneDC (blocked on manual OneDrive download) — user declined to pursue for now. Recommend locking in v17 and shifting effort to 07-18 decoder package deliverables (README, factsheet, submission email) unless a new idea surfaces.

## Code-submission (07-18) prep: standalone decoder built and verified
Caught and fixed a real gap: our submission uses 4 different AEIC checkpoints picked per-image by the
knapsack, but the required test-phase bitstream naming (`bitstream/000001.bin`) has no room for a
checkpoint tag — nothing in the submitted files told a decoder which of the 4 checkpoints to use.
Fixed by writing a 1-byte checkpoint-id header into every bitstream (`refine.py --ckpt_id`, consumed by
`decode_bitstream()`), and building `AEIC/src/decode.py`, a real standalone bitstream->PNG decoder with
no source/GT dependency, that reads the header and dispatches automatically. Verified end-to-end on 2
real val bitstreams via `decode_selfcheck()` — both decoded correctly with the right checkpoint dispatch.
Also fixed `package_decoder()`, which was silently defaulting to the stale v11-era checkpoint set
(`r2b_7000,...`) and the old `enhancer_out_v4` instead of v17's actual lineage (`r2_18000` swap,
`enhancer_out_matched/enhancer_18000.pt`) — would have shipped the wrong decoder if uncaught.
Real package size confirmed: 4.756GB (under the 5GB decoder cap). Note: the email attachment cap is
separately **4GB** (per challenge details.md) — our package exceeds that, so it must go via a download
link, not a direct attachment, when the code-submission email is sent.

## Research notes (OneDC evaluation, pursued 07-14 -- rejected)
Downloaded OneDC (NeurIPS 2025) weights (exlow_bpp0034 ultra-low-bitrate model + VQGAN tokenizer, user
manually pulled from OneDrive after the 403/JS-folder block). Built a separate torch 2.5 Modal image
(modal_onedc.py, OneDC_code/ vendored from github.com/onedc-codec/onedc), compiled their rANS entropy
coder, worked around a gated-repo blocker (stabilityai/stable-diffusion-2-1 now requires auth; patched
to the public sd2-community/stable-diffusion-2-1 mirror), worked around A10G OOM on full-res 2K images
(no tiled decode in OneDC's VAE path, unlike AEIC -- tested on the sub-1K-pixel val images only).

Found the same defect that killed OSCAR (E62): the z_only ultra-low-bitrate inference path computes
`bpp = patches * 14 bits` (fixed-length VQ-index count) rather than invoking their own compiled rANS
coder -- no real serialized bitstream, would need real entropy/fixed-length packing built before this
could be submission-legal.

Quality screen (6 sub-1K val images, real GT comparison): mean score **~82.16 @ 0.0034 bpp** (PSNR
16-20, LPIPS 0.24-0.40, DISTS 0.13-0.23) -- dramatically worse than AEIC's 111.5 @ 0.025 bpp, despite
using ~7x fewer bits. Gap too large to plausibly close by testing the higher-rate lambda checkpoints
(each a separate 4.58GB single-rate download with the same bitstream-realism gap to fix first).
**Rejected.** No remaining untested codec-family lever; v17 (board 31.5270) stands as the practical
ceiling for this architecture.

| E65 | 07-15 | Downsample-encode + SR-decode screen for the 2K tier (6 imgs, LANCZOS 2x down, r2b_7000 encode, Real-ESRGAN x2plus up, scored vs full-res GT at source-pixel bpp) | **96.61 @ 0.0140** vs same-image v17 baseline **108.13 @ 0.0221** — saves 0.008 bpp, loses 11.5 pts; LPIPS 0.247 vs 0.09 (SR hallucination punished by reference metrics) | decisively negative, closed. modal_sr.py kept for reference |

| E66 | 07-15 | TTO through frozen enhancer (--enhancer_ckpt in refine.py loss chain; latent optimized against the actually-scored output), 6-img subset @300 iters vs same-image v17 baseline | mean **−0.036/img**, mixed signs, all within noise | neutral/negative — enhancer is near-identity residual, gradient path through it doesn't move the optimum. Closed. |

| E68 | 07-17 | Conditioning-refresh TTO (--refreeze_every 75: re-derive frozen entropy conditioning from current latent every 75 iters), 8-img screen vs single-freeze | 6/8 improve (+0.10..+0.32), one outlier −1.02, mean −0.003 as replacement — but as an ADDED knapsack candidate losers are discarded: E[max(0,Δ)] ≈ +0.12/img | full 100-img run launched (pkgset_v6_refreeze) — last score experiment before package freeze |
| E69 | 07-17 | Model soup: 0.5·r2_18000 + 0.5·r3l4_8000 weight average, full-val encode | 112.1297 @ 0.0371 vs parents 112.1812/112.2216 — worse than both | negative, closed |

| E70 | 07-18 | E68 full run scored: refreeze TTO (100 imgs) + champion enhancer | **111.6717 @ 0.03101** vs v3_long's 111.5264 @ 0.03115 — **+0.145 raw at same bits** | biggest lever since 300-iter TTO |
| E71 | 07-18 | E67 verdict: TTO-pair enhancer (3k fine-tune steps) on v3_long | 111.518 vs champion's 111.5264 | negative — champion enhancer stays, decoder package unchanged |
| E72 | 07-18 | Exact MILP remix incl. refreeze candidate → **v20, submission_v20_refreeze.zip** | **111.6632 @ w-bpp 0.024990**, refreeze wins 72/100 | board est **31.6632** (+0.136 over v17). Awaiting board confirmation |
