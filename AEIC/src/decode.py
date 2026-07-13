"""Standalone decoder: bitstream -> reconstructed PNG.

Reads the 1-byte checkpoint-tag header this team's encoder (refine.py) writes into every
.bin file, dispatches to the matching AEIC checkpoint, decodes, and applies the fixed
post-decode enhancer (zero additional bits). No access to source/GT images required.

Usage:
python decode.py --sd_path=./sd-turbo --vae_decoder_path=./adcsr/halfDecoder.ckpt \
  --ckpt_dir=./aeic_ckpts --enhancer_ckpt=./enhancer.pt \
  --bin_path=<bitstream_dir> --rec_path=<output_dir>
"""
import argparse, os
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd_path", required=True)
    ap.add_argument("--vae_decoder_path", required=True)
    ap.add_argument("--ckpt_dir", required=True, help="dir containing the 4 shipped AEIC .pkl checkpoints")
    ap.add_argument("--enhancer_ckpt", required=True)
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

    nets = {}  # ckpt_id -> loaded AEIC net, built lazily so we only load checkpoints actually used
    names = sorted(os.listdir(args.bin_path))
    for name in names:
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
            out = enhancer(rec).clamp(0, 1)[0].cpu()
        transforms.ToPILImage()(out).save(os.path.join(args.rec_path, name + ".png"))
        print("[decoded]", name, "ckpt_id", ckpt_id, flush=True)


if __name__ == "__main__":
    main()
