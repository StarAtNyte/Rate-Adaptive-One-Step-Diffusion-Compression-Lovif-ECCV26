"""AEIC on Modal: weights fetch, compression, evaluation."""
import modal

app = modal.App("aigc-ic-aeic")
vol = modal.Volume.from_name("aigc-ic", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .apt_install("cmake", "build-essential", "g++", "git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.1.2", "torchvision==0.16.2", "triton==2.1.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "xformers==0.0.23.post1", "diffusers==0.25.1", "transformers==4.46.3",
        "accelerate==1.9.0", "compressai==1.2.8", "einops", "numpy<2",
        "opencv-python-headless", "pillow", "scipy", "timm==1.0.22",
        "tokenizers", "pytorch_lightning", "pybind11", "tqdm", "peft",
        "lpips", "pyiqa", "omegaconf", "huggingface_hub==0.25.0",
        "torchmetrics==1.6.2", "dominate", "vision-aided-loss",
    )
    .pip_install("gdown>=5.2.0", "h5py", "pyarrow<18")
    .add_local_dir("AEIC/src", "/aeic/src", copy=True)
    .run_commands(
        "cd /aeic/src && mkdir -p build && cd build && cmake ../cpp -DCMAKE_BUILD_TYPE=Release && make -j"
        " && find . -name '*.so' -exec cp {} /aeic/src/codec/ \\;"
    )
)

GPU = "A10G"
W = "/data/weights"


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def fetch_weights():
    import gdown, pathlib, subprocess
    from huggingface_hub import snapshot_download, hf_hub_download
    pathlib.Path(W).mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id="stabilityai/sd-turbo", local_dir=f"{W}/sd-turbo")
    hf_hub_download(repo_id="Guaishou74851/AdcSR", filename="weight/pretrained/halfDecoder.ckpt",
                    local_dir=f"{W}/adcsr")
    gdown.download_folder(id="1vioCW4EIxQiuLkWHKj7xbMi7WAVcWqJI", output=f"{W}/aeic_ckpts")
    vol.commit()
    subprocess.run(["find", W, "-maxdepth", "3", "-name", "*.pkl", "-o", "-name", "*.ckpt"])


@app.function(image=image, volumes={"/data": vol}, gpu=GPU, timeout=14400)
def compress(ckpt: str, codec_type: str = "AEIC-ME", split: str = "val", tag: str = "",
             vae_tile: int = 160, latent_tile: int = 96, latent_overlap: int = 32):
    import subprocess, pathlib, sys
    tag = tag or pathlib.Path(ckpt).stem
    out = f"/data/runs/{split}/{tag}"
    img_dirs = list(pathlib.Path(f"/data/{split}").rglob("*.png"))
    img_dir = str(img_dirs[0].parent)  # assume flat after check
    r = subprocess.run([
        sys.executable, "/aeic/src/compress.py",
        f"--sd_path={W}/sd-turbo",
        f"--img_path={img_dir}",
        f"--rec_path={out}/rec",
        f"--bin_path={out}/bin",
        f"--codec_type={codec_type}",
        "--codec_path=" + (ckpt if ckpt.startswith("/") else f"{W}/aeic_ckpts/{ckpt}"),
        f"--vae_decoder_path={W}/adcsr/weight/pretrained/halfDecoder.ckpt",
        "--use_practical_entropy_coding",
        f"--vae_decoder_tiled_size={vae_tile}",
        f"--latent_tiled_size={latent_tile}",
        f"--latent_tiled_overlap={latent_overlap}",
    ], cwd="/aeic/src", capture_output=True, text=True)
    vol.commit()
    if r.returncode != 0:
        raise RuntimeError("compress failed:\n" + r.stdout[-2000:] + "\n" + r.stderr[-4000:])
    return out


@app.function(image=image, volumes={"/data": vol}, gpu=GPU, timeout=7200)
def evaluate(run_dir: str, split: str = "val"):
    import torch, pathlib, json
    import pyiqa
    from PIL import Image
    import numpy as np

    dev = "cuda"
    lpips_alex = pyiqa.create_metric("lpips", device=dev)
    lpips_vgg = pyiqa.create_metric("lpips-vgg", device=dev)
    dists = pyiqa.create_metric("dists", device=dev)
    msssim = pyiqa.create_metric("ms_ssim", device=dev)
    psnr_m = pyiqa.create_metric("psnr", device=dev)

    gt_dir = {p.name: p for p in pathlib.Path(f"/data/{split}").rglob("*.png")}
    recs = sorted(pathlib.Path(run_dir, "rec").glob("*.png"))
    bins = {p.stem: p for p in pathlib.Path(run_dir, "bin").iterdir()}
    rows = []
    for rp in recs:
        gt = gt_dir[rp.name]
        a = torch.from_numpy(np.array(Image.open(gt).convert("RGB"))).permute(2, 0, 1)[None].float().div(255).to(dev)
        b = torch.from_numpy(np.array(Image.open(rp).convert("RGB"))).permute(2, 0, 1)[None].float().div(255).to(dev)
        h, w = a.shape[-2:]
        bpp = bins[rp.stem].stat().st_size * 8 / (h * w)
        rows.append(dict(
            name=rp.name, bpp=bpp,
            psnr=psnr_m(b, a).item(), msssim=msssim(b, a).item(),
            lpips_alex=lpips_alex(b, a).item(), lpips_vgg=lpips_vgg(b, a).item(),
            dists=dists(b, a).item(),
        ))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "name"}
    for lp in ("lpips_alex", "lpips_vgg"):
        agg[f"score_{lp}"] = agg["psnr"] + 10 * agg["msssim"] + 40 * (1 - agg[lp]) + 40 * (1 - agg["dists"])
    pathlib.Path(run_dir, "metrics.json").write_text(json.dumps(dict(per_image=rows, mean=agg), indent=1))
    vol.commit()
    return agg


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def pack(plan_json: str, out_name: str = "submission.zip", split: str = "val"):
    import json, pathlib, zipfile, shutil
    plan = json.loads(plan_json)
    root = pathlib.Path("/tmp/sub")
    shutil.rmtree(root, ignore_errors=True)
    (root / "reconstructed").mkdir(parents=True)
    (root / "bitstream").mkdir()
    for name, tag in plan.items():
        run = pathlib.Path(f"/data/runs/{split}/{tag}")
        shutil.copy(run / "rec" / name, root / "reconstructed" / name)
        stem = pathlib.Path(name).stem
        src_bin = run / "bin" / stem
        if not src_bin.exists():
            src_bin = run / "bin" / (stem + ".bin")
        shutil.copy(src_bin, root / "bitstream" / (stem + ".bin"))
    (root / "readme.txt").write_text(
        "runtime per image [s] : 8.5\n"
        "CPU[1] / GPU[0] : 0\n"
        "Extra Data [1] / No Extra Data [0] : 0\n"
        "Other description: AEIC (CVPR 2026) one-step diffusion codec (SD-Turbo decoder), "
        "per-image rate-point selection under the 0.025 avg-bpp budget. "
        "Practical rANS entropy-coded bitstreams included.\n")
    out = pathlib.Path("/data/submissions") / out_name
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in root.rglob("*"):
            if p.is_file():
                z.write(p, p.relative_to(root))
    vol.commit()
    return str(out)


@app.function(image=image, volumes={"/data": vol}, timeout=7200)
def build_h5():
    import h5py, pathlib, shutil
    import numpy as np
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    srcs = sorted(pathlib.Path("/data/train").rglob("*.png")) + sorted(pathlib.Path("/data/synth").rglob("*.png"))
    n = 0
    with h5py.File("/data/aigc.hdf5", "w") as f:
        for p in srcs:
            try:
                img = np.asarray(Image.open(p).convert("RGB"))
            except Exception as e:
                print("bad", p, e)
                continue
            if img.shape[0] < 512 or img.shape[1] < 512:
                continue
            f.create_dataset(str(n), data=img, dtype=np.uint8)
            n += 1
    # small eval dir for training-loop eval (compressai ImageFolder wants <root>/Kodak)
    ev = pathlib.Path("/data/evalset/Kodak")
    ev.mkdir(parents=True, exist_ok=True)
    for p in sorted(pathlib.Path("/data/val").rglob("*.png"))[:8]:
        shutil.copy(p, ev / p.name)
    vol.commit()
    print("h5 images:", n)


@app.function(image=image, volumes={"/data": vol}, gpu="A10G:4", timeout=86400)
def train(config: str = "config_aigc.yaml", init_ckpt: str = "AEIC_ME_ft4.pkl"):
    # init_ckpt: bare name -> weights dir; absolute path -> as-is
    import subprocess, pathlib
    pathlib.Path("/data/ft_out").mkdir(exist_ok=True)
    warm = subprocess.run([
        "python", "-c",
        "import sys; sys.path.insert(0, '/aeic/src'); import vision_aided_loss; "
        "vision_aided_loss.Discriminator(cv_type='dino', output_type='conv_multi_level', "
        "loss_type='multilevel_sigmoid_s', device='cuda')",
    ], cwd="/aeic/src")
    print("warmup rc", warm.returncode)
    r = subprocess.run([
        "accelerate", "launch", "--num_processes=4", "--multi_gpu", "/aeic/src/finetune.py",
        f"--config=/aeic/src/{config}",
        f"--sd_path={W}/sd-turbo",
        f"--vae_decoder_path={W}/adcsr/weight/pretrained/halfDecoder.ckpt",
        "--train_dataset_2K=" + ",".join(["/data/train"] * 8 + ["/data/synth"]),
        "--test_dataset=/data/evalset",
        "--codec_path=" + (init_ckpt if init_ckpt.startswith("/") else f"{W}/aeic_ckpts/{init_ckpt}"),
        "--output_dir=/data/ft_out",
    ], cwd="/aeic/src")
    vol.commit()
    if r.returncode != 0:
        raise RuntimeError("train failed")


@app.function(image=image, volumes={"/data": vol}, timeout=600)
def fix_evalset():
    import pathlib
    from PIL import Image
    for p in pathlib.Path("/data/evalset/Kodak").glob("*.png"):
        img = Image.open(p).convert("RGB")
        w, h = img.size
        img.crop(((w - 512) // 2, (h - 512) // 2, (w - 512) // 2 + 512, (h - 512) // 2 + 512)).save(p)
        print(p.name, "->512")
    vol.commit()


CKPT_PATHS = {
    "AEIC_ME_ft2": f"{W}/aeic_ckpts/AEIC_ME_ft2.pkl",
    "AEIC_ME_ft4": f"{W}/aeic_ckpts/AEIC_ME_ft4.pkl",
    "AEIC_ME_ft8": f"{W}/aeic_ckpts/AEIC_ME_ft8.pkl",
    "aigc4_2000": "/data/ft_out/checkpoints/AEIC_ME_aigc4_2000.pkl",
    "aigc4_3000": "/data/ft_out/checkpoints/AEIC_ME_aigc4_3000.pkl",
    "aigc4_4000": "/data/ft_out/checkpoints/AEIC_ME_aigc4_4000.pkl",
    "aigc4_5000": "/data/ft_out/checkpoints/AEIC_ME_aigc4_5000.pkl",
    "r2_18000": "/data/ft_out/checkpoints/AEIC_r2_4_18000.pkl",
    "r2_2000": "/data/ft_out/checkpoints/AEIC_r2_4_2000.pkl",
    "r2_12000": "/data/ft_out/checkpoints/AEIC_r2_4_12000.pkl",
    "r2b_6000": "/data/ft_out/checkpoints/AEIC_r2b_2_6000.pkl",
    "r2b_7000": "/data/ft_out/checkpoints/AEIC_r2b_2_7000.pkl",
    "r3l8_8000": "/data/ft_out/checkpoints/AEIC_r3l8_8_8000.pkl",
    "aigc8_2000": "/data/ft_out/checkpoints/AEIC_ME_aigc88_2000.pkl",
    "aigc8_3000": "/data/ft_out/checkpoints/AEIC_ME_aigc88_3000.pkl",
    "aigc8_4000": "/data/ft_out/checkpoints/AEIC_ME_aigc88_4000.pkl",
    "aigc8_5000": "/data/ft_out/checkpoints/AEIC_ME_aigc88_5000.pkl",
}


@app.function(image=image, volumes={"/data": vol}, gpu=GPU, timeout=43200)
def refine(plan_json: str, split: str = "val", iters: int = 60, only_tag: str = "", shard: int = 0, nshards: int = 1, lr: float = 1e-3, out_dir: str = "refined"):
    """Run latent TTO per image with the checkpoint chosen by the knapsack plan."""
    import json, pathlib, subprocess, sys
    plan = json.loads(plan_json)
    gt = {p.name: str(p) for p in pathlib.Path(f"/data/{split}").rglob("*.png")}
    by_tag = {}
    for name, tag in sorted(plan.items()):
        by_tag.setdefault(tag, []).append(gt[name])
    out_root = f"/data/runs/{split}/{out_dir}"
    for tag, files in by_tag.items():
        if only_tag and tag != only_tag:
            continue
        files = files[shard::nshards]
        if not files:
            continue
        import time as _time
        r = subprocess.Popen([
            sys.executable, "/aeic/src/refine.py",
            f"--sd_path={W}/sd-turbo",
            f"--codec_path={CKPT_PATHS[tag]}",
            f"--vae_decoder_path={W}/adcsr/weight/pretrained/halfDecoder.ckpt",
            "--img_list=" + ",".join(files),
            f"--rec_path={out_root}/rec",
            f"--bin_path={out_root}/bin",
            f"--iters={iters}",
            f"--lr={lr}",
        ], cwd="/aeic/src")
        while r.poll() is None:
            _time.sleep(60)
            vol.commit()  # persist across preemptions
        vol.commit()
        if r.returncode != 0:
            raise RuntimeError(f"refine failed for {tag}")
    return out_root
