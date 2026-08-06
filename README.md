# Rate-Adaptive One-Step Diffusion Compression

This repository contains the public implementation and paper source for our
challenge submission on learned compression of AI-generated images. The method
uses a rate-adaptive selection of independently decodable reconstructions and
combines a shallow learned codec with one-step diffusion restoration.

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

## License

The bundled AEIC implementation is released under the MIT License; see
[`AEIC/LICENSE`](AEIC/LICENSE). Check the licenses of external checkpoints,
datasets, and third-party dependencies before redistributing them.
