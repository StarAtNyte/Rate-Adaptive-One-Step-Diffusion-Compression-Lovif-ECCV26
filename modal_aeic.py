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
    img_dirs = [p for p in pathlib.Path(f"/data/{split}").rglob("*.png") if "__MACOSX" not in p.parts]
    if not img_dirs:
        raise RuntimeError(f"no images found under /data/{split}")
    dir_counts = {}
    for p in img_dirs:
        dir_counts[p.parent] = dir_counts.get(p.parent, 0) + 1
    img_dir = str(max(dir_counts, key=dir_counts.get))  # dir with the most images
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
    "r3l4_8000": "/data/ft_out/checkpoints/AEIC_r3l4_4_8000.pkl",
}


@app.function(image=image, volumes={"/data": vol}, gpu=GPU, timeout=43200)
def refine(plan_json: str, split: str = "val", iters: int = 60, only_tag: str = "", shard: int = 0, nshards: int = 1, lr: float = 1e-3, out_dir: str = "refined",
           text_boxes_path: str = "", text_bias: float = 0.7, text_dists_w: float = 3.0, only_with_text: bool = False):
    """Run latent TTO per image with the checkpoint chosen by the knapsack plan."""
    import json, pathlib, subprocess, sys
    plan = json.loads(plan_json)
    gt = {p.name: str(p) for p in pathlib.Path(f"/data/{split}").rglob("*.png")}
    text_boxes = {}
    if text_boxes_path:
        text_boxes = json.loads(pathlib.Path(text_boxes_path).read_text())
    by_tag = {}
    for name, tag in sorted(plan.items()):
        if only_with_text and not text_boxes.get(name):
            continue
        by_tag.setdefault(tag, []).append(gt[name])
    out_root = f"/data/runs/{split}/{out_dir}"
    for tag, files in by_tag.items():
        if only_tag and tag != only_tag:
            continue
        files = files[shard::nshards]
        if not files:
            continue
        import time as _time
        cmd = [
            sys.executable, "/aeic/src/refine.py",
            f"--sd_path={W}/sd-turbo",
            f"--codec_path={CKPT_PATHS[tag]}",
            f"--vae_decoder_path={W}/adcsr/weight/pretrained/halfDecoder.ckpt",
            "--img_list=" + ",".join(files),
            f"--rec_path={out_root}/rec",
            f"--bin_path={out_root}/bin",
            f"--iters={iters}",
            f"--lr={lr}",
        ]
        if text_boxes_path:
            cmd += [f"--text_boxes_json={text_boxes_path}", f"--text_bias={text_bias}", f"--text_dists_w={text_dists_w}"]
        r = subprocess.Popen(cmd, cwd="/aeic/src")
        while r.poll() is None:
            _time.sleep(60)
            vol.commit()  # persist across preemptions
        vol.commit()
        if r.returncode != 0:
            raise RuntimeError(f"refine failed for {tag}")
    return out_root


diag_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch==2.1.2", "torchvision==0.16.2", extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("easyocr", "pyiqa", "pillow", "numpy<2", "opencv-python-headless")
)


@app.function(image=diag_image, volumes={"/data": vol}, gpu="A10G", timeout=3600)
def text_diagnostic(plan_json: str, split: str = "val", pad: int = 8):
    """OCR-detect text regions on GT, compare text-crop vs whole-image LPIPS/DISTS
    between GT and the plan's chosen reconstruction, to test if text drags down score."""
    import json, pathlib, easyocr, numpy as np, torch, pyiqa
    from PIL import Image

    plan = json.loads(plan_json)
    reader = easyocr.Reader(["en"], gpu=True)
    dev = "cuda"
    lpips_m = pyiqa.create_metric("lpips", device=dev)
    dists_m = pyiqa.create_metric("dists", device=dev)

    gt_dir = {p.name: p for p in pathlib.Path(f"/data/{split}").rglob("*.png")}
    rows = []
    for name, tag in sorted(plan.items()):
        gt_path = gt_dir[name]
        rec_path = pathlib.Path(f"/data/runs/{split}/{tag}/rec/{name}")
        if not rec_path.exists():
            continue
        gt_img = Image.open(gt_path).convert("RGB")
        rec_img = Image.open(rec_path).convert("RGB")
        gt_np = np.array(gt_img)
        H, W = gt_np.shape[:2]

        boxes = reader.readtext(gt_np, detail=1, paragraph=False)
        text_boxes = []
        for bbox, txt, conf in boxes:
            if conf < 0.3 or not txt.strip():
                continue
            xs = [p[0] for p in bbox]; ys = [p[1] for p in bbox]
            x0, x1 = max(0, int(min(xs)) - pad), min(W, int(max(xs)) + pad)
            y0, y1 = max(0, int(min(ys)) - pad), min(H, int(max(ys)) + pad)
            if x1 - x0 < 32:
                cx = (x0 + x1) // 2
                x0, x1 = max(0, cx - 16), min(W, cx + 16)
            if y1 - y0 < 32:
                cy = (y0 + y1) // 2
                y0, y1 = max(0, cy - 16), min(H, cy + 16)
            if x1 - x0 < 32 or y1 - y0 < 32:
                continue
            text_boxes.append((x0, y0, x1, y1))

        def to_t(img):
            return torch.from_numpy(np.array(img)).permute(2, 0, 1)[None].float().div(255).to(dev)

        whole_lpips = lpips_m(to_t(rec_img), to_t(gt_img)).item()
        whole_dists = dists_m(to_t(rec_img), to_t(gt_img)).item()

        text_lpips = text_dists = None
        if text_boxes:
            tl, td, area = 0.0, 0.0, 0
            for x0, y0, x1, y1 in text_boxes:
                gt_c = gt_img.crop((x0, y0, x1, y1))
                rec_c = rec_img.crop((x0, y0, x1, y1))
                w = (x1 - x0) * (y1 - y0)
                tl += lpips_m(to_t(rec_c), to_t(gt_c)).item() * w
                td += dists_m(to_t(rec_c), to_t(gt_c)).item() * w
                area += w
            text_lpips, text_dists = tl / area, td / area

        rows.append(dict(name=name, tag=tag, n_text_boxes=len(text_boxes),
                          whole_lpips=whole_lpips, whole_dists=whole_dists,
                          text_lpips=text_lpips, text_dists=text_dists))
        print(name, "boxes:", len(text_boxes), "whole_lpips:", round(whole_lpips, 4),
              "text_lpips:", round(text_lpips, 4) if text_lpips else None, flush=True)

    with_text = [r for r in rows if r["n_text_boxes"] > 0]
    if with_text:
        mean_whole_l = sum(r["whole_lpips"] for r in with_text) / len(with_text)
        mean_text_l = sum(r["text_lpips"] for r in with_text) / len(with_text)
        mean_whole_d = sum(r["whole_dists"] for r in with_text) / len(with_text)
        mean_text_d = sum(r["text_dists"] for r in with_text) / len(with_text)
        print(f"\n=== SUMMARY: {len(with_text)}/{len(rows)} images have text ===")
        print(f"LPIPS  whole={mean_whole_l:.4f}  text-region={mean_text_l:.4f}  delta={mean_text_l-mean_whole_l:+.4f}")
        print(f"DISTS  whole={mean_whole_d:.4f}  text-region={mean_text_d:.4f}  delta={mean_text_d-mean_whole_d:+.4f}")

    out = pathlib.Path(f"/data/runs/{split}/text_diagnostic.json")
    out.write_text(json.dumps(rows, indent=1))
    vol.commit()
    return str(out)


@app.function(image=diag_image, volumes={"/data": vol}, gpu="A10G", timeout=1800)
def ocr_boxes(split: str = "val", pad: int = 8):
    """Detect text boxes on every GT image, save {name: [[x0,y0,x1,y1],...]} for reuse."""
    import json, pathlib, easyocr, numpy as np
    from PIL import Image

    reader = easyocr.Reader(["en"], gpu=True)
    out = {}
    for p in sorted(pathlib.Path(f"/data/{split}").rglob("*.png")):
        img = np.array(Image.open(p).convert("RGB"))
        H, W = img.shape[:2]
        boxes = reader.readtext(img, detail=1, paragraph=False)
        kept = []
        for bbox, txt, conf in boxes:
            if conf < 0.3 or not txt.strip():
                continue
            xs = [pt[0] for pt in bbox]; ys = [pt[1] for pt in bbox]
            x0, x1 = max(0, int(min(xs)) - pad), min(W, int(max(xs)) + pad)
            y0, y1 = max(0, int(min(ys)) - pad), min(H, int(max(ys)) + pad)
            if x1 - x0 < 32:
                cx = (x0 + x1) // 2; x0, x1 = max(0, cx - 16), min(W, cx + 16)
            if y1 - y0 < 32:
                cy = (y0 + y1) // 2; y0, y1 = max(0, cy - 16), min(H, cy + 16)
            if x1 - x0 >= 32 and y1 - y0 >= 32:
                kept.append([x0, y0, x1, y1])
        out[p.name] = kept
        print(p.name, len(kept), flush=True)
    dest = pathlib.Path(f"/data/{split}_text_boxes.json")
    dest.write_text(json.dumps(out, indent=1))
    vol.commit()
    return str(dest)


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", timeout=43200)
def train_enhancer(rec_dirs: str, gt_dir: str = "/data/train/train", out_dir: str = "/data/enhancer_out",
                    steps: int = 8000, resume: str = "", n_blocks: int = 8,
                    ea_dists_w: float = 1.0, gan_w: float = 0.0, lr: float = 2e-4):
    import subprocess, sys
    cmd = [
        sys.executable, "/aeic/src/enhancer.py",
        f"--rec_dirs={rec_dirs}",
        f"--gt_dir={gt_dir}",
        f"--out_dir={out_dir}",
        f"--steps={steps}",
        f"--n_blocks={n_blocks}",
        f"--ea_dists_w={ea_dists_w}",
        f"--gan_w={gan_w}",
        f"--lr={lr}",
    ]
    if resume:
        cmd.append(f"--resume={resume}")
    import time as _time
    r = subprocess.Popen(cmd, cwd="/aeic/src")
    while r.poll() is None:
        _time.sleep(60)
        vol.commit()
    vol.commit()
    if r.returncode != 0:
        raise RuntimeError("train_enhancer failed")
    return out_dir


@app.function(image=image, volumes={"/data": vol}, gpu=GPU, timeout=7200)
def apply_enhancer(enhancer_ckpt: str, run_dir: str, out_dir: str):
    import sys, pathlib, torch
    sys.path.insert(0, "/aeic/src")
    from enhancer import Enhancer
    from PIL import Image
    from torchvision import transforms

    net = Enhancer().cuda().eval()
    net.load_state_dict(torch.load(enhancer_ckpt, map_location="cuda"))
    tf = transforms.ToTensor()

    src = pathlib.Path(run_dir)
    dst = pathlib.Path(out_dir)
    (dst / "rec").mkdir(parents=True, exist_ok=True)
    (dst / "bin").mkdir(parents=True, exist_ok=True)
    for p in sorted((src / "rec").glob("*.png")):
        img = tf(Image.open(p).convert("RGB")).cuda().unsqueeze(0)
        with torch.no_grad():
            out = net(img).clamp(0, 1)[0].cpu()
        transforms.ToPILImage()(out).save(dst / "rec" / p.name)
    for p in (src / "bin").iterdir():
        (dst / "bin" / p.name).write_bytes(p.read_bytes())
    vol.commit()
    return str(dst)
