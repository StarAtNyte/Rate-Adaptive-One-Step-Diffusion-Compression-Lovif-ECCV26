"""Downsample-encode / SR-decode prototype (screen for the 2K tier).
Upscales AEIC reconstructions of 2x-downsampled images back to source resolution with
Real-ESRGAN x2plus, then scores vs the ORIGINAL full-res GT with bits/(source pixels).
"""
import modal

app = modal.App("sr-screen")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0", "wget")
    .pip_install("torch==2.1.2", "torchvision==0.16.2",
                 extra_index_url="https://download.pytorch.org/whl/cu121")
    .pip_install("pyiqa", "pillow", "numpy<2", "opencv-python-headless", "timm==1.0.22")
    .run_commands("mkdir -p /weights && wget -q -O /weights/RealESRGAN_x2plus.pth "
                  "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth")
)

vol = modal.Volume.from_name("aigc-ic")


def build_rrdbnet():
    """RRDBNet(x2) exactly as basicsr defines it, inlined to avoid the basicsr dependency."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb3(self.rdb2(self.rdb1(x)))
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, scale=2, num_feat=64, num_block=23, num_grow_ch=32):
            super().__init__()
            self.scale = scale
            if scale == 2:
                num_in_ch = num_in_ch * 4
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, True)

        def forward(self, x):
            if self.scale == 2:
                feat = F.pixel_unshuffle(x, downscale_factor=2)
            else:
                feat = x
            feat = self.conv_first(feat)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out

    return RRDBNet()


@app.function(image=image, volumes={"/data": vol}, gpu="A10G", timeout=1800)
def sr_and_eval(rec_dir: str, bin_dir: str, gt_split: str = "val", out_dir: str = ""):
    """Upscale rec_dir 2x with Real-ESRGAN, score vs full-res GT, bpp = bin bits / SOURCE pixels."""
    import torch, pathlib, json
    import numpy as np
    import pyiqa
    from PIL import Image

    dev = "cuda"
    net = build_rrdbnet().to(dev).eval()
    sd = torch.load("/weights/RealESRGAN_x2plus.pth", map_location=dev)
    net.load_state_dict(sd["params_ema"] if "params_ema" in sd else sd["params"], strict=True)

    lpips_alex = pyiqa.create_metric("lpips", device=dev)
    dists = pyiqa.create_metric("dists", device=dev)
    msssim = pyiqa.create_metric("ms_ssim", device=dev)
    psnr_m = pyiqa.create_metric("psnr", device=dev)

    gt_dir = {p.name: p for p in pathlib.Path(f"/data/{gt_split}").rglob("*.png") if "__MACOSX" not in p.parts}
    out_path = pathlib.Path(out_dir) if out_dir else None
    if out_path:
        (out_path / "rec").mkdir(parents=True, exist_ok=True)

    rows = []
    for rp in sorted(pathlib.Path(rec_dir).glob("*.png")):
        gt = Image.open(gt_dir[rp.name]).convert("RGB")
        W, H = gt.size
        lr = Image.open(rp).convert("RGB")
        x = torch.from_numpy(np.array(lr)).permute(2, 0, 1)[None].float().div(255).to(dev)
        with torch.no_grad():
            y = net(x).clamp(0, 1)
        y = y[:, :, :H, :W]
        if out_path:
            from torchvision.transforms import ToPILImage
            ToPILImage()(y[0].cpu()).save(out_path / "rec" / rp.name)

        a = torch.from_numpy(np.array(gt)).permute(2, 0, 1)[None].float().div(255).to(dev)
        stem = rp.stem
        bp = pathlib.Path(bin_dir) / stem
        if not bp.exists():
            bp = bp.with_suffix(".bin")
        bpp = bp.stat().st_size * 8 / (H * W)
        row = dict(name=rp.name, bpp=bpp,
                   psnr=psnr_m(y, a).item(), msssim=msssim(y, a).item(),
                   lpips_alex=lpips_alex(y, a).item(), dists=dists(y, a).item())
        row["score"] = row["psnr"] + 10 * row["msssim"] + 40 * (1 - row["lpips_alex"]) + 40 * (1 - row["dists"])
        rows.append(row)
        print(row, flush=True)

    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "name"}
    print("MEAN:", json.dumps(agg, indent=1))
    if out_path:
        (out_path / "metrics_sr.json").write_text(json.dumps(dict(per_image=rows, mean=agg), indent=1))
        vol.commit()
    return agg
