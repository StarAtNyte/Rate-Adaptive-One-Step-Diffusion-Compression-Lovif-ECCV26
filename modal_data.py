import modal

app = modal.App("aigc-ic-data")
vol = modal.Volume.from_name("aigc-ic", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("gdown>=5.2.0").apt_install("unzip", "file")

FILES = {
    "train.zip": "1i7DPC9qlhc9dw8TqnfmCxhVJaPmqw0VA",
    "val.zip": "1ufJyoUM_9ic3YIxICeImNEF1GkfQjCY6",
}


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def download():
    import gdown, pathlib, subprocess
    print("gdown", gdown.__version__)
    for name, fid in FILES.items():
        out = pathlib.Path("/data/raw") / name
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            print("skip", name)
            continue
        gdown.download(id=fid, output=str(out))
        subprocess.run(["file", str(out)])
    vol.commit()
    for p in pathlib.Path("/data/raw").iterdir():
        print(p, p.stat().st_size)


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def extract():
    import subprocess, pathlib
    for name in FILES:
        src = f"/data/raw/{name}"
        dst = f"/data/{name.removesuffix('.zip')}"
        if not pathlib.Path(dst).exists():
            subprocess.run(["unzip", "-q", src, "-d", dst], check=True)
    vol.commit()
    for d in pathlib.Path("/data").iterdir():
        n = sum(1 for _ in d.rglob("*.png"))
        print(d, "pngs:", n)
