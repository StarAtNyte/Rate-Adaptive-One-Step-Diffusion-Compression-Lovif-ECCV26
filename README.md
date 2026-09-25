# Rate-Adaptive One-Step Diffusion Compression for AIGC Images (LoViF, ECCV 2026)

This repository contains the public implementation and paper source for our
entry to the **LoViF 2026 AIGC Image Compression Challenge** — ultra-low-bitrate
coding of AI-generated images under a strict global rate budget of 0.025 bits
per pixel (BPP). Our system fine-tunes four rate-specialized checkpoints of the
AEIC one-step diffusion codec, generates per-image candidates (including one
refined through encoder-side test-time optimization), entropy-codes every
candidate with practical rANS coding, and selects exactly one bitstream per
image via a multiple-choice-knapsack allocation over true coded file sizes. A
fixed, zero-additional-bit residual restoration network runs at decode time.
Every submitted bitstream is independently decodable with no source image or
external side information required.

## Paper

- **Challenge paper (OpenReview, official challenge track):**
  [openreview.net/forum?id=41hfEpLx7Z](https://openreview.net/challenge?redirect=%2Fforum%3Fid%3D41hfEpLx7Z)
- **Personal OpenReview listing:**
  [openreview.net/forum?id=yaLXjEBS7v](https://openreview.net/forum?id=yaLXjEBS7v&noteId=yaLXjEBS7v)
- PDF and LaTeX source are also bundled in this repo under [`paper/`](paper/).

## Result

Our entry scored **31.527739** (PSNR 27.02 dB, MS-SSIM 0.9176, LPIPS 0.0778,
DISTS 0.0390 at 0.02495 BPP) — the second-best DISTS on the leaderboard,
ranking **5th** on the organizer-recalculated final test-phase leaderboard
announced August 4, 2026.

## Repository contents

- [`AEIC/`](AEIC/) — the AEIC codec and training/inference utilities used by the
  compression pipeline.
- [`paper/`](paper/) —
  LaTeX source, bibliography, figures, and the final paper PDF.

## Reproduction

The AEIC subproject documents its environment, data preparation, external model
downloads, and inference commands in [`AEIC/README.md`](AEIC/README.md). A full
reproduction requires those external datasets/checkpoints and a CUDA-capable
PyTorch environment; this repository does not claim one-command reproduction of
the submitted challenge archive.

## Citation

```bibtex
@inproceedings{khanal2026rateadaptive,
  title     = {Rate-Adaptive One-Step Diffusion Compression for {AIGC} Images},
  author    = {Khanal, Nitiz},
  booktitle = {LoViF 2026 AIGC Image Compression Challenge, ECCV 2026 Workshop},
  year      = {2026},
  url       = {https://openreview.net/forum?id=yaLXjEBS7v}
}
```

## License

The bundled AEIC implementation is released under the MIT License; see
[`AEIC/LICENSE`](AEIC/LICENSE). Check the licenses of external checkpoints,
datasets, and third-party dependencies before redistributing them.
