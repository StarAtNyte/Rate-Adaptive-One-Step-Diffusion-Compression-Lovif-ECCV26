"""Decoder-side post-enhancer: small residual CNN, restores AEIC output toward GT.

Ships inside the submitted decoder (applied after AEIC decode). Trained on
(reconstruction, GT) pairs from train-800 across multiple rate points, with
LPIPS+DISTS loss -- the same metrics the challenge scores on. Never touches val.

Architecture: lightweight RRDB-style residual body, no resolution change,
output = input + residual (identity-safe init via zero-init final conv).
"""
import argparse, glob, os, random, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import lpips as lpips_pkg
import pyiqa
import vision_aided_loss


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


def sobel_edges(x):
    """x: (B,3,H,W) in [0,1]. Returns per-channel edge magnitude."""
    gray = (0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3])
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    gx = F.conv2d(gray, kx, padding=1)
    gy = F.conv2d(gray, ky, padding=1)
    edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    return edge.repeat(1, 3, 1, 1)  # 3ch so DISTS (expects RGB) is happy


class Enhancer(nn.Module):
    def __init__(self, ch=64, n_blocks=8):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(n_blocks)])
        self.tail = nn.Conv2d(ch, 3, 3, 1, 1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        f = self.head(x)
        f = self.body(f)
        return x + self.tail(f)  # residual, identity at init


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, rec_dirs, gt_dir, patch=384):
        self.pairs = []
        for rec_dir in rec_dirs:
            for p in sorted(glob.glob(os.path.join(rec_dir, "*.png"))):
                gt_p = os.path.join(gt_dir, os.path.basename(p))
                if os.path.exists(gt_p):
                    self.pairs.append((p, gt_p))
        self.patch = patch
        self.tf = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rec_p, gt_p = self.pairs[idx]
        rec = Image.open(rec_p).convert("RGB")
        gt = Image.open(gt_p).convert("RGB")
        w, h = gt.size
        ps = min(self.patch, w, h)
        x0 = random.randint(0, w - ps)
        y0 = random.randint(0, h - ps)
        rec = rec.crop((x0, y0, x0 + ps, y0 + ps))
        gt = gt.crop((x0, y0, x0 + ps, y0 + ps))
        if random.random() < 0.5:
            rec = rec.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        return self.tf(rec), self.tf(gt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec_dirs", required=True, help="comma-separated rec dirs")
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--patch", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--l1_w", type=float, default=1.0)
    ap.add_argument("--lpips_w", type=float, default=1.0)
    ap.add_argument("--dists_w", type=float, default=1.5)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--resume", default="")
    ap.add_argument("--n_blocks", type=int, default=8)
    ap.add_argument("--ea_dists_w", type=float, default=1.0, help="edge-aware DISTS term weight (0 disables)")
    ap.add_argument("--gan_w", type=float, default=0.0, help="adversarial loss weight (0 disables GAN)")
    ap.add_argument("--disc_lr", type=float, default=2e-5)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ds = PairDataset(args.rec_dirs.split(","), args.gt_dir, patch=args.patch)
    print(f"dataset: {len(ds)} pairs", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                     num_workers=4, drop_last=True, persistent_workers=True)

    net = Enhancer(n_blocks=args.n_blocks).cuda()
    if args.resume:
        net.load_state_dict(torch.load(args.resume, map_location="cuda"))
        print("resumed from", args.resume, flush=True)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.05)

    lpips_loss = lpips_pkg.LPIPS(net="alex").cuda()
    lpips_loss.requires_grad_(False)
    dists_loss = pyiqa.create_metric("dists", device="cuda", as_loss=True)

    net_disc = opt_disc = None
    if args.gan_w > 0:
        net_disc = vision_aided_loss.Discriminator(cv_type="dino", output_type="conv_multi_level",
                                                     loss_type="multilevel_sigmoid_s", device="cuda")
        net_disc = net_disc.cuda()
        net_disc.cv_ensemble.requires_grad_(False)
        net_disc.train()
        opt_disc = torch.optim.AdamW(net_disc.parameters(), lr=args.disc_lr)

    step = 0
    t0 = time.time()
    while step < args.steps:
        for rec, gt in dl:
            if step >= args.steps:
                break
            rec, gt = rec.cuda(), gt.cuda()
            out = net(rec).clamp(0, 1)
            l1 = F.l1_loss(out, gt)
            lp = lpips_loss(out * 2 - 1, gt * 2 - 1).mean()
            dt = dists_loss(out, gt).mean()
            ea = torch.tensor(0.0, device=out.device)
            if args.ea_dists_w > 0:
                ea = dists_loss(sobel_edges(out), sobel_edges(gt)).mean()
            gan_g = torch.tensor(0.0, device=out.device)
            if net_disc is not None:
                gan_g = net_disc(out * 2 - 1, for_G=True).mean()
            loss = (args.l1_w * l1 + args.lpips_w * lp + args.dists_w * dt
                    + args.ea_dists_w * ea + args.gan_w * gan_g)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

            if net_disc is not None:
                d_real = net_disc(gt.detach() * 2 - 1, for_real=True).mean()
                opt_disc.zero_grad(); d_real.backward(); opt_disc.step()
                d_fake = net_disc(out.detach() * 2 - 1, for_real=False).mean()
                opt_disc.zero_grad(); d_fake.backward(); opt_disc.step()

            sched.step()
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{args.steps} loss={loss.item():.4f} l1={l1.item():.4f} "
                      f"lpips={lp.item():.4f} dists={dt.item():.4f} ea={ea.item():.4f} gan_g={gan_g.item():.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
            if step % args.ckpt_every == 0 or step == args.steps:
                torch.save(net.state_dict(), os.path.join(args.out_dir, f"enhancer_{step}.pt"))
                print("saved", step, flush=True)


if __name__ == "__main__":
    main()
