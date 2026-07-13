"""Screen OSCAR on the challenge validation set using its published checkpoints."""
import modal

app = modal.App("aigc-ic-oscar")
vol = modal.Volume.from_name("aigc-ic", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "g++")
    .pip_install(
        "torch==2.0.1", "torchvision==0.15.2", "xformers==0.0.20",
        extra_index_url="https://download.pytorch.org/whl/cu118",
    )
    .pip_install(
        "diffusers==0.25.0", "transformers==4.37.2", "accelerate==1.3.0",
        "peft==0.9.0", "compressai==1.2.8", "vector-quantize-pytorch==1.22.3",
        "huggingface_hub==0.25.0", "numpy==1.26.3", "pillow", "einops", "einx==0.3.0",
        "opencv-python-headless", "tqdm", "safetensors",
    )
    .add_local_dir("OSCAR", "/oscar", copy=True)
)

W = "/data/weights/oscar"


@app.function(image=image, volumes={"/data": vol}, timeout=7200)
def fetch_weights():
    from huggingface_hub import snapshot_download, hf_hub_download
    import pathlib
    pathlib.Path(W).mkdir(parents=True, exist_ok=True)
    snapshot_download(
        "SfinOe/stable-diffusion-v2-1", local_dir=f"{W}/sd21",
        allow_patterns=["vae/*", "unet/*", "scheduler/*", "model_index.json"],
    )
    hf_hub_download("jinpeig/OSCAR", "oscar.pkl", local_dir=W)
    vol.commit()


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", timeout=14400)
def screen(split: str = "val", levels: str = "0,1,2", limit: int = 12, seed: int = 42):
    """Reconstruct a representative prefix; bins encode the published nominal rate.

    This is a quality gate, not yet a submission artifact. Real VQ-index packing is
    implemented only if this screen is competitive.
    """
    import sys, pathlib, math, json, random
    from types import SimpleNamespace
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image

    sys.path.insert(0, "/oscar")
    sys.path.insert(0, "/oscar/diffusion")
    from diffusion.oscar import OSCAR

    args = SimpleNamespace(
        pretrained_model_name_or_path=f"{W}/sd21", lora_rank=16,
        hyper_dim=320, vae_decoder_tiled_size=224,
        vae_encoder_tiled_size=1024, latent_tiled_size=96,
        latent_tiled_overlap=32, merge_and_unload_lora=False,
    )
    torch.manual_seed(seed)
    model = OSCAR(args).cuda().eval()
    model.set_eval()
    model.load_ckpt(torch.load(f"{W}/oscar.pkl", map_location="cpu"))
    model.vae.tile_sample_min_size = 512
    model.vae.tile_latent_min_size = 64
    model.vae.enable_tiling()
    model.unet.enable_xformers_memory_efficient_attention()

    paths = sorted(p for p in pathlib.Path(f"/data/{split}").rglob("*.png") if "__MACOSX" not in p.parts)
    if limit:
        paths = paths[:limit]
    done = {}
    for level in [int(x) for x in levels.split(",")]:
        tag = f"oscar_l{level}_n{len(paths)}"
        root = pathlib.Path(f"/data/runs/{split}/{tag}")
        (root / "rec").mkdir(parents=True, exist_ok=True)
        (root / "bin").mkdir(parents=True, exist_ok=True)
        for p in paths:
            im = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0
            x = torch.from_numpy(im).permute(2, 0, 1)[None].cuda() * 2 - 1
            h, w = x.shape[-2:]
            ph, pw = (-h) % 128, (-w) % 128
            xpad = F.pad(x, (0, pw, 0, ph), mode="reflect")
            # The published encoder samples the VAE posterior; reset per image so
            # results are reproducible and independent of traversal order.
            torch.manual_seed(seed)
            with torch.inference_mode():
                y, _, _ = model(xpad, level)
            y = y[..., :h, :w].float().clamp(-1, 1).add(1).div(2)
            out = (y[0].permute(1, 2, 0).cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
            Image.fromarray(out).save(root / "rec" / p.name)
            # Screening proxy: exact byte count corresponding to OSCAR's declared bpp.
            nbytes = math.ceil(model.bpps[level] * h * w / 8)
            (root / "bin" / f"{p.stem}.bin").write_bytes(bytes(nbytes))
        done[level] = str(root)
    (pathlib.Path(f"/data/runs/{split}") / "oscar_screen.json").write_text(json.dumps(done, indent=2))
    vol.commit()
    return done
