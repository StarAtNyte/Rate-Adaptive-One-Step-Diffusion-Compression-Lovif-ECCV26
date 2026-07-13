"""Per-image encoder-side latent refinement (test-time optimization).

Optimizes codec latent y through the FROZEN decoder chain (codec cond-decode ->
one-step UNet -> VAE) against the challenge score surrogate, then writes a real
bitstream via codec.compress(y_override=y*). Decoder untouched -> submission-legal.

Usage:
python refine.py --sd_path ... --codec_path ... --vae_decoder_path ... \
  --img_list img1.png,img2.png --rec_path out/rec --bin_path out/bin \
  --iters 60 --codec_type AEIC-ME
"""
import os, math, time, argparse, glob
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from AEIC_practical import AEIC
from codec.codec_practical import ste_round
from my_utils.compress_utils import my_write_body, my_read_body, write_uchars, read_uchars, filesize, Path
import lpips as lpips_pkg
import pyiqa


def forward_ste(codec, y, x_pad):
    """Replicates PixelCodec.inference but differentiable w.r.t. y (STE rounding)."""
    y_h, y_w = x_pad.shape[2] // 32, x_pad.shape[3] // 32
    z_h, z_w = math.ceil(y_h / 4), math.ceil(y_w / 4)
    pad_h, pad_w = z_h * 4 - y_h, z_w * 4 - y_w

    y_padded = F.pad(y, pad=(0, pad_w, 0, pad_h), mode='constant')
    z = codec.h_a(y_padded)
    _, qz_lik = codec.entropy_bottleneck(z, training=False)
    z_offset = codec.entropy_bottleneck._get_medians()
    z_hat = ste_round(z - z_offset) + z_offset

    B, C, H, W = y.shape
    masks = codec.get_mask_four_parts(B, C, H, W, device=y.device)
    base = codec.h_s(z_hat)[:, :, 0:y_h, 0:y_w]
    scales_sum, means_sum = 0, 0
    y_hat_last = None
    for i in range(4):
        mask = masks[i]
        means_supp, scales_supp = codec.adapter_out[i](codec.g_c(codec.adapter_in[i](base))).chunk(2, 1)
        means = means_supp * mask
        scales = scales_supp * mask
        y_i = y * mask
        y_hat_i = ste_round(y_i - means) + means
        base = base * (1 - mask) + y_hat_i
        scales_sum = scales_sum + scales
        means_sum = means_sum + means
        y_hat_last = y_hat_i

    _, qy_lik = codec.gaussian_conditional(y, scales_sum, means_sum, training=False)
    y_hat = base  # after last group merge, base == full y_hat

    num_pixels = x_pad.shape[2] * x_pad.shape[3]
    bpp = (torch.log(qy_lik).sum() + torch.log(qz_lik).sum()) / (-math.log(2) * num_pixels)
    return y_hat, bpp


def get_frozen(codec, y, x_pad):
    """Precompute detached conditioning (means/scales/z-rate) from initial y."""
    y_h, y_w = x_pad.shape[2] // 32, x_pad.shape[3] // 32
    z_h, z_w = math.ceil(y_h / 4), math.ceil(y_w / 4)
    pad_h, pad_w = z_h * 4 - y_h, z_w * 4 - y_w
    y_padded = F.pad(y, pad=(0, pad_w, 0, pad_h), mode='constant')
    z = codec.h_a(y_padded)
    _, qz_lik = codec.entropy_bottleneck(z, training=False)
    z_offset = codec.entropy_bottleneck._get_medians()
    z_hat = torch.round(z - z_offset) + z_offset
    B, C, H, W = y.shape
    masks = codec.get_mask_four_parts(B, C, H, W, device=y.device)
    base = codec.h_s(z_hat)[:, :, 0:y_h, 0:y_w]
    scales_sum, means_sum = 0, 0
    for i in range(4):
        mask = masks[i]
        means_supp, scales_supp = codec.adapter_out[i](codec.g_c(codec.adapter_in[i](base))).chunk(2, 1)
        means = means_supp * mask
        scales = scales_supp * mask
        y_hat_i = torch.round(y * mask - means) + means
        base = base * (1 - mask) + y_hat_i
        scales_sum = scales_sum + scales
        means_sum = means_sum + means
    num_pixels = x_pad.shape[2] * x_pad.shape[3]
    z_bpp = torch.log(qz_lik).sum() / (-math.log(2) * num_pixels)
    return dict(means=means_sum.detach(), scales=scales_sum.detach(),
                z_bpp=z_bpp.detach(), num_pixels=num_pixels)


def forward_frozen(codec, y, frozen):
    """Differentiable w.r.t. y only; conditioning frozen."""
    y_hat = ste_round(y - frozen["means"]) + frozen["means"]
    _, qy_lik = codec.gaussian_conditional(y, frozen["scales"], frozen["means"], training=False)
    bpp = torch.log(qy_lik).sum() / (-math.log(2) * frozen["num_pixels"]) + frozen["z_bpp"]
    return y_hat, bpp


def unet_fwd(net, x):
    # grad-enabled single-tile unet (refine crops/images are decoded whole when they fit;
    # large latents fall back to net.unet directly - memory permitting)
    net.forbid_tiled_unet = True
    try:
        return net.unet(x)
    finally:
        net.forbid_tiled_unet = False


def decode_bitstream(net, bin_path, name):
    """Reads the 1-byte checkpoint-tag header (see write side in main()) then the body.
    Caller is responsible for having loaded `net` with the checkpoint that byte identifies —
    see CKPT_ID_TO_TAG in decode.py for the id<->checkpoint mapping."""
    output = os.path.join(bin_path, name)
    with Path(output).open("rb") as f:
        read_uchars(f, 1)  # ckpt_id header, consumed by the caller before dispatch
        strings, shape = my_read_body(f)
    ori_h, ori_w = shape
    padded_y_h = math.ceil(ori_h / 64) * 2
    padded_y_w = math.ceil(ori_w / 64) * 2
    z_size = (1, net.codec.y_channel // 2, math.ceil(padded_y_h / 4), math.ceil(padded_y_w / 4))
    net.codec.entropy_coder.reset()
    net.codec.entropy_coder.set_stream(strings)
    out_img, _ = net.decompress(z_size, padded_y_h, padded_y_w)
    out_img = out_img[:, :, 0:ori_h, 0:ori_w]
    return (out_img * 0.5 + 0.5).float().cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sd_path", required=True)
    ap.add_argument("--codec_path", required=True)
    ap.add_argument("--vae_decoder_path", required=True)
    ap.add_argument("--codec_type", default="AEIC-ME")
    ap.add_argument("--img_list", required=True, help="comma-separated png paths")
    ap.add_argument("--rec_path", required=True)
    ap.add_argument("--bin_path", required=True)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rate_w", type=float, default=20000)
    ap.add_argument("--mse_w", type=float, default=1000)
    ap.add_argument("--dists_w", type=float, default=40)
    ap.add_argument("--lora_rank_unet", default=32, type=int)
    ap.add_argument("--enable_xformers_memory_efficient_attention", default=True)
    ap.add_argument("--color_fix", default=True)
    ap.add_argument("--merge_LoRA", default=True)
    ap.add_argument("--compile_model", default=False)
    ap.add_argument("--use_tiled_vae", default=True)
    ap.add_argument("--use_tiled_unet", default=True)
    ap.add_argument("--vae_decoder_tiled_size", type=int, default=160)
    ap.add_argument("--latent_tiled_size", type=int, default=96)
    ap.add_argument("--latent_tiled_overlap", type=int, default=32)
    ap.add_argument("--use_practical_entropy_coding", default=True, action="store_true")
    ap.add_argument("--crop_ly", type=int, default=16, help="TTO crop size in latent px (16=512px, 24=768px, 32=1024px)")
    ap.add_argument("--text_boxes_json", default="", help="path to {name: [[x0,y0,x1,y1],...]} pixel-coord boxes")
    ap.add_argument("--text_bias", type=float, default=0.7, help="fraction of iters that sample a text-box crop when available")
    ap.add_argument("--text_dists_w", type=float, default=3.0, help="extra DISTS weight multiplier on text crops")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for per-iteration crop sampling")
    ap.add_argument("--ckpt_id", type=int, default=0, help="1-byte tag written into the bitstream header so a "
                     "standalone decoder can tell which of the shipped checkpoints to load (see decode.py)")
    args = ap.parse_args()

    text_boxes_all = {}
    if args.text_boxes_json:
        import json as _json
        text_boxes_all = _json.loads(open(args.text_boxes_json).read())

    net = AEIC(sd_path=args.sd_path, args=args)
    net.cuda().eval()
    net.requires_grad_(False)
    net.codec.update(force=True)

    lpips_loss = lpips_pkg.LPIPS(net='alex').cuda()
    lpips_loss.requires_grad_(False)
    dists_loss = pyiqa.create_metric("dists", device="cuda", as_loss=True)

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.5] * 3, [0.5] * 3)])
    os.makedirs(args.rec_path, exist_ok=True)
    os.makedirs(args.bin_path, exist_ok=True)

    for img_path in args.img_list.split(","):
        t0 = time.time()
        name = os.path.splitext(os.path.basename(img_path))[0]
        if os.path.exists(os.path.join(args.rec_path, name + ".png")) and os.path.exists(os.path.join(args.bin_path, name)):
            print("[skip existing]", name, flush=True)
            continue
        img = tf(Image.open(img_path).convert("RGB")).cuda().unsqueeze(0)
        ori_h, ori_w = img.shape[2:]
        pad_h = math.ceil(ori_h / 64) * 64 - ori_h
        pad_w = math.ceil(ori_w / 64) * 64 - ori_w
        x = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')

        with torch.no_grad():
            y0 = net.codec.g_a(x)
            frozen = get_frozen(net.codec, y0, x)
            _, bpp0 = forward_frozen(net.codec, y0, frozen)
        y = y0.clone().requires_grad_(True)
        opt = torch.optim.Adam([y], lr=args.lr)

        import random
        rng = random.Random(args.seed)
        yH, yW = y0.shape[2:]
        LY = min(args.crop_ly, yH, yW)
        M = 2                  # decode margin in y-latent px
        img_boxes = text_boxes_all.get(name + ".png", []) or text_boxes_all.get(name, [])

        for it in range(args.iters):
          if True:
            y_hat, bpp = forward_frozen(net.codec, y, frozen)
            is_text_iter = bool(img_boxes) and rng.random() < args.text_bias
            if is_text_iter:
                bx0, by0, bx1, by1 = img_boxes[rng.randrange(len(img_boxes))]
                bca, bcb = (by0 + by1) // 64, (bx0 + bx1) // 64  # px center -> latent idx (//32) then centered
                a = max(0, min(yH - LY, bca - LY // 2))
                b = max(0, min(yW - LY, bcb - LY // 2))
            else:
                a = rng.randint(0, max(0, yH - LY)) if yH > LY else 0
                b = rng.randint(0, max(0, yW - LY)) if yW > LY else 0
            a0, a1 = max(0, a - M), min(yH, a + LY + M)
            b0, b1 = max(0, b - M), min(yW, b + LY + M)
            if (a1 - a0) % 2:
                if a1 - 1 >= a + LY: a1 -= 1
                else: a0 += 1
            if (b1 - b0) % 2:
                if b1 - 1 >= b + LY: b1 -= 1
                else: b0 += 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l_T, res1 = net.codec.g_s(y_hat[:, :, a0:a1, b0:b1])
                l_0 = unet_fwd(net, l_T) + res1
                xc_hat = net.vae.decoder(l_0).clamp(-1, 1)
            xc_hat = xc_hat.float()
            # trim decode margin, map latent window -> pixels (x32)
            ta, tb = (a - a0) * 32, (b - b0) * 32
            xc_hat = xc_hat[:, :, ta:ta + LY * 32, tb:tb + LY * 32]
            xc = x[:, :, a * 32:(a + LY) * 32, b * 32:(b + LY) * 32]
            x01, xh01 = xc * 0.5 + 0.5, xc_hat * 0.5 + 0.5
            mse = F.mse_loss(xh01, x01)
            lp = lpips_loss(xc_hat, xc).mean()
            dt = dists_loss(xh01, x01).mean()
            dt_w = args.dists_w * (args.text_dists_w if is_text_iter else 1.0)
            rate_pen = F.relu(bpp - bpp0.detach())
            loss = args.mse_w * mse + 40 * lp + dt_w * dt + args.rate_w * rate_pen
          if True:
            opt.zero_grad()
            loss.backward()
            opt.step()
            if it % 20 == 0:
                print(f"{name} it{it} loss {loss.item():.3f} mse {mse.item():.5f} "
                      f"lpips {lp.item():.4f} dists {dt.item():.4f} bpp {bpp.item():.5f}", flush=True)

        with torch.no_grad():
            net.codec.entropy_coder.reset()
            net.codec.compress(x, y_override=y.detach())
            strings = net.codec.entropy_coder.get_encoded_stream()
            outb = os.path.join(args.bin_path, name)
            with Path(outb).open("wb") as f:
                write_uchars(f, [args.ckpt_id])
                my_write_body(f, [ori_h, ori_w], strings)
            out_img = decode_bitstream(net, args.bin_path, name)
            bpp_real = filesize(outb) * 8 / (ori_h * ori_w)
        transforms.ToPILImage()(out_img[0].clamp(0, 1)).save(os.path.join(args.rec_path, name + ".png"))
        print(f"[done] {name} bpp {bpp_real:.5f} time {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
