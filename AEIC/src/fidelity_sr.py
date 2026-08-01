"""Fidelity-first learned decoder for the reduced-resolution entropy codec."""
import argparse
import glob
import os
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_tensor


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return x + 0.1 * self.body(x)


class FidelitySR(nn.Module):
    """Identity-safe residual restoration; works at arbitrary image sizes."""
    def __init__(self, channels=64, blocks=12):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, 1, 1)
        self.body = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, 1, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        return x + self.tail(self.body(self.head(x)))


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, rec_dir, gt_dir, patch=256):
        gt = {os.path.basename(p): p for p in glob.glob(os.path.join(gt_dir, "**", "*.png"), recursive=True)}
        self.pairs = [(p, gt[os.path.basename(p)]) for p in glob.glob(os.path.join(rec_dir, "*.png"))
                      if os.path.basename(p) in gt]
        self.patch = patch

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        rec = Image.open(self.pairs[index][0]).convert("RGB")
        gt = Image.open(self.pairs[index][1]).convert("RGB")
        w, h = gt.size
        p = min(self.patch, w, h)
        x = random.randrange(w - p + 1)
        y = random.randrange(h - p + 1)
        rec, gt = rec.crop((x, y, x+p, y+p)), gt.crop((x, y, x+p, y+p))
        if random.random() < .5:
            rec, gt = rec.transpose(Image.Transpose.FLIP_LEFT_RIGHT), gt.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        return to_tensor(rec), to_tensor(gt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rec_dir", required=True)
    p.add_argument("--gt_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--patch", type=int, default=192)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--blocks", type=int, default=12)
    p.add_argument("--resume", default="")
    p.add_argument("--lpips_w", type=float, default=.03)
    p.add_argument("--lpips_every", type=int, default=4)
    p.add_argument("--checkpoint_every", type=int, default=250)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    ds = PairDataset(args.rec_dir, args.gt_dir, args.patch)
    if not ds.pairs:
        raise RuntimeError("no reconstruction/GT pairs found")
    print("pairs", len(ds), flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                     drop_last=True, persistent_workers=True)
    net = FidelitySR(blocks=args.blocks).cuda()
    if args.resume:
        saved = torch.load(args.resume, map_location="cuda")
        net.load_state_dict(saved["model"] if "model" in saved else saved)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=args.lr/20)
    perceptual = None
    if args.lpips_w:
        import lpips
        perceptual = lpips.LPIPS(net="alex").cuda().eval().requires_grad_(False)
    step, start = 0, time.time()
    while step < args.steps:
        for rec, gt in dl:
            rec, gt = rec.cuda(), gt.cuda()
            out = net(rec).clamp(0, 1)
            mse = F.mse_loss(out, gt)
            # MSE dominates because every +dB directly increases challenge score.
            loss = mse
            lp = torch.zeros((), device="cuda")
            if perceptual is not None and step % args.lpips_every == 0:
                lp = perceptual(out*2-1, gt*2-1).mean()
                # Preserve the expected perceptual gradient while avoiding an
                # expensive feature-network pass on every MSE update.
                loss = loss + args.lpips_w * args.lpips_every * lp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                psnr = -10 * torch.log10(mse.detach()).item()
                print(f"step {step} loss={loss.item():.6f} psnr={psnr:.3f} lpips={lp.item():.4f} t={time.time()-start:.0f}", flush=True)
            if step % args.checkpoint_every == 0 or step == args.steps:
                torch.save({"model": net.state_dict(), "blocks": args.blocks},
                           os.path.join(args.out_dir, f"fidelity_sr_{step}.pt"))
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
