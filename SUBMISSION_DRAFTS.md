# 07-18 Code Submission — drafts (fill team details before sending)

## Email

**To:** xin.li@ustc.edu.cn
**Subject:** [AIGCIC2026] - [AIGC Image Compression] - [ZeroR]

Body:

> Full challenge name: LoViF 2026 - AIGC Image Compression Challenge
> Team name: ZeroR
> Team leader's name and email address: Nitiz Khanal, khanalnitij20@gmail.com
> Alternative email: 077bel025.nitiz@pcampus.edu.np
> Affiliation: Pulchowk Campus, Lalitpur, Nepal
> Team name and usernames registered on the LoViF 2026 Codabench server: ZeroR / nitizkhanal
>
> Decoder package (4.40 GB zip, exceeds the 4 GB attachment cap, provided as a download link per the
> submission guidelines): <FILL: upload decoder_package_v17.zip to Drive and paste share link>
>
> The package contains all trained decoder models and parameters (trimmed fp16 SD-Turbo UNet+VAE,
> AdcSR halfDecoder, 4 fine-tuned AEIC-ME checkpoints, post-decode enhancer) plus inference code and
> a README describing how to run the standalone decoder (src/decode.py: bitstream dir -> reconstructed
> PNGs, no source-image access; each bitstream carries a 1-byte header selecting the checkpoint).
> Total unpacked size: 4.756 GB (under the 5 GB decoder limit).

## Factsheet skeleton (final due 07-23 with test results; template PDF from organizers — map these in)

- **Method name:** AEIC-ME fine-tuned + per-image test-time latent optimization + decoder-side enhancer
- **Base:** AEIC (CVPR 2026, Zhang et al., MIT license), SD-Turbo one-step diffusion decoder, ME variant
- **Training data:** provided train-800 + self-generated synthetic AIGC corpus (~15.7k images; prompts
  from GenEval / DPG-Bench / CVTG-2K / LongText-Bench; generators SDXL, SD3.5-medium, PixArt-Sigma,
  Sana). All public. No test/val data used in training. (Extra Data: 1 — disclose corpus.)
- **Pipeline (encode):** per-image knapsack selects one of 4 rate-tier checkpoints under the global
  0.025 weighted-bpp budget; 300-step Adam latent TTO against LPIPS+DISTS+MSE with frozen
  autoregressive entropy conditioning; real rANS bitstreams; 1-byte checkpoint-id bitstream header.
- **Pipeline (decode):** AEIC decode (tiled VAE) -> fixed residual CNN enhancer (8-block RRDB-ish,
  1.5M params, zero bitstream cost).
- **Runtime:** encode ~8.5 s/image (A10G, TTO-dominated); decode ~1–2 s/image GPU.
- **Reproducibility:** private repo StarAtNyte/lovif-aigc-compression (LOGS.md = full experiment
  ledger E1–E66); decoder package is self-contained.
- **Board result (dev):** 31.5270 (submission_v17_shippable.zip), rank 2 at time of writing.

## Checklist before sending
- [ ] Upload decoder_package_v17.zip somewhere link-shareable (Drive), test the link in incognito
- [ ] Fill team leader name + Codabench username
- [ ] If the long-shot enhancer retrain (E67, running) wins: rebuild package with the new enhancer
      via `modal run modal_aeic.py::package_decoder --enhancer-path <new>` and re-upload — decision
      deadline is package upload time, not experiment convenience
