"""Phase B: synthetic AIGC training corpus on Modal.

Prompts: GenEval + DPG-Bench + CVTG-2K + LongText-Bench (same sources as challenge data).
Models: FLUX.1-schnell, SDXL-base, PixArt-Sigma, Sana (all non-gated).
"""
import modal

app = modal.App("aigc-ic-gen")
vol = modal.Volume.from_name("aigc-ic", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "torchvision", extra_index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("diffusers==0.32.2", "transformers==4.48.0", "accelerate", "sentencepiece",
                 "datasets", "pillow", "protobuf", "ftfy", "beautifulsoup4")
)

PROMPTS = "/data/prompts.jsonl"


@app.function(image=image, volumes={"/data": vol}, timeout=1800)
def build_prompts():
    import json, urllib.request, pathlib
    rows = []

    # GenEval
    url = "https://raw.githubusercontent.com/djghosh13/geneval/main/prompts/evaluation_metadata.jsonl"
    for line in urllib.request.urlopen(url).read().decode().splitlines():
        rows.append({"src": "geneval", "prompt": json.loads(line)["prompt"]})

    # DPG-Bench (prompts dir of ELLA repo)
    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as td:
        subprocess.run(["pip", "install", "-q", "huggingface_hub"], check=False)
        from huggingface_hub import snapshot_download
        try:
            d = snapshot_download(repo_id="jieliu/DPG-Bench-Prompts", repo_type="dataset", local_dir=td)
            for p in pathlib.Path(d).rglob("*.txt"):
                rows.append({"src": "dpg", "prompt": p.read_text().strip()})
        except Exception as e:
            print("DPG via HF failed:", e)
            import io, zipfile
            z = urllib.request.urlopen("https://github.com/TencentQQGYLab/ELLA/archive/refs/heads/main.zip").read()
            zf = zipfile.ZipFile(io.BytesIO(z))
            for n in zf.namelist():
                if "dpg_bench/prompts/" in n and n.endswith(".txt"):
                    rows.append({"src": "dpg", "prompt": zf.read(n).decode().strip()})

    # CVTG-2K: zip of json dicts {idx: prompt}
    import io, zipfile
    z = urllib.request.urlopen(
        "https://huggingface.co/datasets/dnkdnk/CVTG-2K/resolve/main/CVTG-2K.zip").read()
    zf = zipfile.ZipFile(io.BytesIO(z))
    for n in zf.namelist():
        if n.endswith("_combined.json"):
            for p in json.loads(zf.read(n)).values():
                rows.append({"src": "cvtg", "prompt": p})

    # LongText-Bench
    from datasets import load_dataset
    try:
        ds = load_dataset("X-Omni/LongText-Bench")
        split = list(ds.values())[0]
        key = next(k for k in split.column_names if "prompt" in k.lower() or "text" in k.lower() or "caption" in k.lower())
        for r in split:
            rows.append({"src": "longtext", "prompt": str(r[key])})
    except Exception as e:
        print("longtext failed:", e)

    rows = [r for r in rows if r["prompt"]]
    import pathlib as pl
    pl.Path(PROMPTS).write_text("\n".join(json.dumps(r) for r in rows))
    vol.commit()
    from collections import Counter
    print(Counter(r["src"] for r in rows), "total", len(rows))


MODELS = {
    "flux": ("black-forest-labs/FLUX.1-schnell", 4),
    "sdxl": ("stabilityai/stable-diffusion-xl-base-1.0", 30),
    "pixart": ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", 20),
    "sana": ("Efficient-Large-Model/Sana_1600M_1024px_diffusers", 20),
}
# ponytail: fixed aspect cycle instead of per-image sampling
SIZES = [(1024, 1024), (1152, 896), (896, 1152), (2048, 2048), (1536, 1024), (1024, 1536)]


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", timeout=86400,
              secrets=[modal.Secret.from_name("huggingface")])
def generate(model_key: str, shard: int = 0, nshards: int = 1, seed: int = 0, srcs: str = ""):
    import json, pathlib, torch
    repo, steps = MODELS[model_key]
    rows = [json.loads(l) for l in pathlib.Path(PROMPTS).read_text().splitlines()]
    rows = [r for r in rows if not srcs or r["src"] in srcs.split(",")]
    rows = rows[shard::nshards]
    outdir = pathlib.Path(f"/data/synth/{model_key}")
    outdir.mkdir(parents=True, exist_ok=True)

    from diffusers import DiffusionPipeline
    pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=torch.bfloat16)
    if model_key == "flux":
        pipe.enable_model_cpu_offload()  # ponytail: flux > 24GB, offload on A10G
        pipe.vae.enable_tiling()
    else:
        pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for i, r in enumerate(rows):
        idx = shard + i * nshards
        out = outdir / f"{idx:06d}_s{seed}.png"
        if out.exists():
            continue
        w, h = SIZES[idx % len(SIZES)]
        if model_key == "flux" and max(w, h) > 1152:
            w, h = min(w, 1152), min(h, 1152)
        if model_key in ("sdxl", "pixart", "sana") and max(w, h) > 1536:
            w, h = w // 2 * 1, h // 2 * 1  # ponytail: non-flux models unreliable at 2K
            w, h = min(w, 1536), min(h, 1536)
        g = torch.Generator("cuda").manual_seed(seed * 100003 + idx)
        try:
            img = pipe(prompt=r["prompt"][:900], width=w, height=h,
                       num_inference_steps=steps, generator=g).images[0]
            img.save(out)
        except Exception as e:
            print("skip", idx, e)
            torch.cuda.empty_cache()
        if i % 50 == 0:
            vol.commit()
            print(f"{model_key} shard{shard}: {i}/{len(rows)}")
    vol.commit()
    return f"{model_key} shard {shard} done"
