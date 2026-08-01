"""Standalone decoder: bitstream -> reconstructed PNG.

Reads the 1-byte checkpoint-tag header this team's encoder (refine.py) writes into every
.bin file, dispatches to the matching AEIC checkpoint, decodes, and applies the fixed
post-decode enhancer (zero additional bits). No access to source/GT images required.

Usage:
python decode.py --sd_path=./sd-turbo --vae_decoder_path=./adcsr/halfDecoder.ckpt \
  --ckpt_dir=./aeic_ckpts --enhancer_ckpt=./enhancer.pt \
  --bin_path=<bitstream_dir> --rec_path=<output_dir>
"""
import argparse, os, struct, zlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from AEIC_practical import AEIC
from refine import decode_bitstream
from my_utils.compress_utils import read_uchars, Path

# id<->checkpoint-file mapping, must match SHIPPABLE_CKPT_ORDER / ckpt_paths in modal_aeic.py
CKPT_ID_TO_FILE = {
    0: "AEIC_ME_aigc88_5000.pkl",
    1: "AEIC_r2_4_18000.pkl",
    2: "AEIC_r3l4_4_8000.pkl",
    3: "AEIC_r3l8_8_8000.pkl",
}


class ResBlock(nn.Module):
    def __init__(self, ch, growth=32):
        super().__init__()
        self.c1 = nn.Conv2d(ch, growth, 3, 1, 1)
        self.c2 = nn.Conv2d(ch + growth, growth, 3, 1, 1)
        self.c3 = nn.Conv2d(ch + 2 * growth, ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        f1 = self.act(self.c1(x))
        f2 = self.act(self.c2(torch.cat([x, f1], 1)))
        f3 = self.c3(torch.cat([x, f1, f2], 1))
        return x + 0.2 * f3


class Enhancer(nn.Module):
    def __init__(self, ch=64, n_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, x):
        f = self.head(x)
        f = self.body(f)
        return x + self.tail(f)


def peek_ckpt_id(bin_path, name):
    with Path(os.path.join(bin_path, name)).open("rb") as f:
        return read_uchars(f, 1)[0]


def residual_sideinfo(bin_path, name, device):
    """Read the optional RSI1 trailer without affecting the AEIC stream parser."""
    data = Path(os.path.join(bin_path, name)).read_bytes()
    if len(data) < 16:
        return None
    magic, gh, gw, qstep, payload_len = struct.unpack(">4sHHfI", data[-16:])
    if magic != b"RSI1" or payload_len > len(data) - 16:
        return None
    raw = zlib.decompress(data[-16-payload_len:-16])
    expected = 3 * gh * gw
    if len(raw) != expected:
        raise RuntimeError(f"bad RSI1 payload in {name}: {len(raw)} != {expected}")
    q = torch.frombuffer(bytearray(raw), dtype=torch.int8).reshape(1, 3, gh, gw)
    return q.to(device=device, dtype=torch.float32) * qstep


def decode_chg1(path, codec, sr):
    """Decode our self-delimiting reduced-resolution CompressAI stream."""
    data = Path(path).read_bytes()
    magic, h, w, lh, lw, ph, pw, sh, sw = struct.unpack(">4s8I", data[:36])
    if magic != b"CHG1":
        raise RuntimeError("not a CHG1 stream")
    pos = 36
    count = struct.unpack(">I", data[pos:pos+4])[0]; pos += 4
    strings = []
    for _ in range(count):
        n = struct.unpack(">I", data[pos:pos+4])[0]; pos += 4
        strings.append(data[pos:pos+n]); pos += n
    if len(strings) != 2:
        raise RuntimeError(f"unsupported CHG1 stream count: {len(strings)}")
    low = codec.decompress([[strings[0]], [strings[1]]], (sh, sw))["x_hat"]
    low = torch.nan_to_num(low[..., :lh, :lw], nan=0.5, posinf=1.0, neginf=0.0).clamp(0, 1)
    rec = F.interpolate(low, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    return sr(rec).clamp(0, 1)


def enhance_tiled(enhancer, x, tile=512, pad=48):
    """Applies `enhancer` (a purely local conv net, no norm layers) tile-by-tile so peak
    activation memory stays bounded regardless of image size. Each patch is padded with real
    image context (clamped at the border) beyond the enhancer's receptive field (~26px), so the
    cropped output is numerically identical to a single full-image forward pass."""
    _, _, h, w = x.shape
    if h <= tile and w <= tile:
        return enhancer(x)
    out = torch.empty_like(x)
    for y0 in range(0, h, tile):
        y1 = min(y0 + tile, h)
        py0, py1 = max(0, y0 - pad), min(h, y1 + pad)
        for x0 in range(0, w, tile):
            x1 = min(x0 + tile, w)
            px0, px1 = max(0, x0 - pad), min(w, x1 + pad)
            patch = enhancer(x[:, :, py0:py1, px0:px1])
            out[:, :, y0:y1, x0:x1] = patch[:, :, y0 - py0:y0 - py0 + (y1 - y0), x0 - px0:x0 - px0 + (x1 - x0)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd_path", required=True)
    ap.add_argument("--vae_decoder_path", required=True)
    ap.add_argument("--ckpt_dir", required=True, help="dir containing the 4 shipped AEIC .pkl checkpoints")
    ap.add_argument("--enhancer_ckpt", required=True)
    ap.add_argument("--fidelity_codec_ckpt", default="")
    ap.add_argument("--fidelity_sr_ckpt", default="")
    ap.add_argument("--bin_path", required=True)
    ap.add_argument("--rec_path", required=True)
    ap.add_argument("--codec_type", default="AEIC-ME")
    ap.add_argument("--use_tiled_vae", default=True)
    ap.add_argument("--use_tiled_unet", default=True)
    ap.add_argument("--vae_decoder_tiled_size", type=int, default=160)
    ap.add_argument("--latent_tiled_size", type=int, default=96)
    ap.add_argument("--latent_tiled_overlap", type=int, default=32)
    ap.add_argument("--merge_LoRA", default=True)
    ap.add_argument("--compile_model", default=False)
    ap.add_argument("--lora_rank_unet", default=32, type=int)
    ap.add_argument("--enable_xformers_memory_efficient_attention", default=True)
    ap.add_argument("--color_fix", default=True)
    ap.add_argument("--use_practical_entropy_coding", default=True, action="store_true")
    args = ap.parse_args()

    os.makedirs(args.rec_path, exist_ok=True)

    enhancer = Enhancer().cuda().eval()
    enhancer.load_state_dict(torch.load(args.enhancer_ckpt, map_location="cuda"))

    fidelity_codec = fidelity_sr = None
    def get_fidelity_models():
        nonlocal fidelity_codec, fidelity_sr
        if fidelity_codec is None:
            from compressai.zoo import mbt2018_mean
            from fidelity_sr import FidelitySR
            fidelity_codec = mbt2018_mean(quality=1, pretrained=False).cuda().eval()
            fidelity_codec.load_state_dict(torch.load(args.fidelity_codec_ckpt, map_location="cuda"))
            fidelity_codec.update(force=True)
            state = torch.load(args.fidelity_sr_ckpt, map_location="cuda")
            fidelity_sr = FidelitySR(blocks=state.get("blocks", 12)).cuda().eval()
            fidelity_sr.load_state_dict(state["model"] if "model" in state else state)
        return fidelity_codec, fidelity_sr

    nets = {}  # ckpt_id -> loaded AEIC net, built lazily so we only load checkpoints actually used
    names = sorted(os.listdir(args.bin_path))
    for name in names:
        path = os.path.join(args.bin_path, name)
        if Path(path).read_bytes()[:4] == b"CHG1":
            codec, sr = get_fidelity_models()
            with torch.no_grad():
                out = decode_chg1(path, codec, sr)[0].cpu()
            import numpy as np
            from PIL import Image
            arr = (out.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(args.rec_path, name + ".png"))
            del out
            torch.cuda.empty_cache()
            print("[decoded]", name, "CHG1", flush=True)
            continue
        ckpt_id = peek_ckpt_id(args.bin_path, name)
        if ckpt_id not in nets:
            args.codec_path = os.path.join(args.ckpt_dir, CKPT_ID_TO_FILE[ckpt_id])
            net = AEIC(sd_path=args.sd_path, args=args)
            net.cuda().eval()
            net.requires_grad_(False)
            net.codec.update(force=True)
            nets[ckpt_id] = net
        with torch.no_grad():
            rec = decode_bitstream(nets[ckpt_id], args.bin_path, name).cuda()
            corr = residual_sideinfo(args.bin_path, name, rec.device)
            if corr is not None:
                # Encoder generated RSI reconstruction on CPU; use the same
                # interpolation kernel/device for exact round-to-byte parity.
                corr = F.interpolate(corr.cpu(), size=rec.shape[-2:], mode="bilinear", align_corners=False)
                # RSI candidates were optimized against the raw AEIC reconstruction.
                rec8 = torch.round(rec.cpu() * 255) / 255
                out = (rec8 + corr).clamp(0, 1)[0].cpu()
            else:
                out = enhance_tiled(enhancer, rec).clamp(0, 1)[0].cpu()
        if corr is not None:
            # Match RSI encoder-side PNG conversion (round-to-nearest).
            import numpy as np
            from PIL import Image
            arr = (out.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(arr).save(os.path.join(args.rec_path, name + ".png"))
        else:
            transforms.ToPILImage()(out).save(os.path.join(args.rec_path, name + ".png"))
        del rec, out
        torch.cuda.empty_cache()
        print("[decoded]", name, "ckpt_id", ckpt_id, flush=True)


if __name__ == "__main__":
    main()
