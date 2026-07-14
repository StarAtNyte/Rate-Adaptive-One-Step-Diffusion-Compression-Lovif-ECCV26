"""OneDC (NeurIPS 2025) integration screen -- second codec family for the knapsack.
Separate Modal app/image from modal_aeic.py: OneDC needs torch 2.5 (vs AEIC's pinned 2.1.2),
so it can't share the AEIC image. See LOGS.md E62 (OSCAR, rejected) for the sibling screen.
"""
import modal

app = modal.App("onedc-screen")

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("cmake", "build-essential", "g++", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.5.0", "torchvision==0.20.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "accelerate==1.2.1", "cycler==0.12.1", "datasets==3.2.0", "diffusers==0.32.1",
        "DISTS_pytorch==0.1", "einops==0.8.0", "huggingface-hub==0.27.1", "imageio==2.36.1",
        "lpips==0.1.4", "matplotlib==3.10.0", "numpy==1.26.4", "omegaconf==2.3.0",
        "openpyxl==3.1.5", "packaging==24.2", "pandas==2.2.3", "peft==0.14.0", "piq==0.8.0",
        "pycocotools==2.0.10", "pytorch-msssim==1.0.0", "safetensors==0.5.2",
        "tensorboard==2.18.0", "tensorboard-data-server==0.7.2", "tensorboardx==2.6.2.2",
        "tokenizers==0.21.0", "torchmetrics==1.8.2", "tqdm==4.67.1", "transformers==4.48.0",
        "vector-quantize-pytorch==1.21.2", "wandb==0.19.3", "pillow",
    )
    .add_local_dir("OneDC_code/src", "/onedc/src", copy=True)
    .run_commands(
        # upstream CMakeLists uses -Werror; our compiler/lib combo trips warnings AEIC hit too
        "cd /onedc/src/cpp && sed -i 's/-Wall -Wextra -pedantic -Werror/-Wall/' CMakeLists.txt"
        " && mkdir -p /onedc/src/build && cd /onedc/src/build"
        " && cmake ../cpp -DCMAKE_BUILD_TYPE=Release && make -j"
        " && find . -name '*.so' -exec cp {} /onedc/src/ \\;"
    )
)

vol = modal.Volume.from_name("aigc-ic")
GPU = "A10G"


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def push_weights():
    """One-time: copy the locally-downloaded OneDC weights (already on this machine
    under OneDC/extracted/) up to the shared volume so GPU functions can read them."""
    import pathlib, shutil
    dst = pathlib.Path("/data/onedc_weights")
    dst.mkdir(exist_ok=True)
    print("run this via `modal volume put`, not push_weights() -- kept as a no-op placeholder")


@app.function(image=image, volumes={"/data": vol}, gpu=GPU, timeout=3600)
def inference(eval_image_path: str = "/data/val_subset", output_path: str = "/data/runs/onedc/bpp0034",
              checkpoint_path: str = "/data/onedc_weights/bpp0034"):
    import subprocess, sys
    r = subprocess.run([
        sys.executable, "models/sd15_onedc_codec_z_only/inference.py",
        "--config_path=config_inference.yaml",
        f"--checkpoint_path={checkpoint_path}",
        f"--eval_image_path={eval_image_path}",
        f"--output_path={output_path}",
    ], cwd="/onedc/src", capture_output=True, text=True)
    print(r.stdout[-6000:])
    if r.returncode != 0:
        raise RuntimeError("inference failed:\n" + r.stderr[-6000:])
    vol.commit()
    return output_path


@app.function(image=image, timeout=60)
def selftest():
    import torch
    print("torch", torch.__version__, "cuda", torch.cuda.is_available())
