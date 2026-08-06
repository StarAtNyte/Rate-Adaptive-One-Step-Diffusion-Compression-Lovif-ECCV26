# AEIC backbone

This directory contains the AEIC learned-compression backbone used and adapted
by the challenge submission in the parent repository. The submission-specific
pipeline, fine-tuning, restoration, test-time optimization, bitstream
packaging, and rate allocation are described in the root project and [`paper/`](../paper/).

## Installation

```bash
conda create -n aeic python=3.10
conda activate aeic
pip install -r requirements.txt
```

The codec requires a CUDA-capable PyTorch environment and external model/data
downloads. See the scripts and configuration files in this directory for the
training and inference entry points.

## Attribution and license

AEIC is retained here under its original MIT license; see [`LICENSE`](LICENSE).
Please preserve the original attribution when redistributing this code and
check the terms of any external checkpoints or datasets.
