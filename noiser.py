import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# compute ramp
def _ramp(step: float, ramp: float) -> float:
    return min(step / ramp, 1.0) if ramp > 0 else 1.0

# coordinates grid for blur kernel
def _coords_grid(N: int, device, dtype):
    ax = torch.arange(N, device=device, dtype=dtype) - 0.5 * (N - 1)
    yy, xx = torch.meshgrid(ax, ax, indexing='ij')
    return torch.stack([yy, xx], dim=-1)  # (N,N,2)

# random blur kernel
def _random_blur_kernel(N=7, device='cpu', dtype=torch.float32):
    t = random.random()
    coords = _coords_grid(N, device, dtype)  # (N,N,2)
    manhat = coords.abs().sum(dim=-1)       # (N,N)

    if t < 0.25:
        sig_gauss = random.uniform(1.0, 3.0)
        r2 = (coords**2).sum(dim=-1)
        vals = torch.exp(-r2 / (2.0 * sig_gauss**2))
    elif t < 0.5:
        theta = random.uniform(0.0, 2.0 * math.pi)
        v = torch.tensor([math.cos(theta), math.sin(theta)], device=device, dtype=dtype)
        dists = (coords * v).sum(dim=-1)
        sig_line = random.uniform(0.25, 1.0)
        w_line = random.uniform(3.0, 0.5 * (N - 1) + 0.1)
        vals = torch.exp(-dists**2 / (2.0 * sig_line**2)) * (manhat < w_line).to(dtype)
    else:
        vals = torch.zeros((N, N), device=device, dtype=dtype)
        vals[N // 2, N // 2] = 1.0

    v = vals / (vals.sum() + 1e-12)
    return v  # (N,N)

# random blur application
def _depthwise_blur_rgb(x: torch.Tensor, k2d: torch.Tensor) -> torch.Tensor:
    # x: (1,3,H,W), k2d: (k,k)
    k = k2d.shape[0]
    w = k2d.view(1, 1, k, k).repeat(3, 1, 1, 1)
    pad = k // 2
    return F.conv2d(x, w, padding=pad, groups=3)

# RGB <-> YCbCr
def _rgb_to_ycbcr_jpeg(x: torch.Tensor) -> torch.Tensor:
    # x: (B,3,H,W), domain [0,255]
    M = torch.tensor([[0.299, 0.587, 0.114],
                      [-0.168736, -0.331264, 0.5],
                      [0.5, -0.418688, -0.081312]], dtype=x.dtype, device=x.device)  # (3,3)
    shift = torch.tensor([0., 128., 128.], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    M_T = M.t()
    y = torch.einsum('bchw,cd->bdhw', x, M_T) + shift
    return y

# YCbCr -> RGB
def _ycbcr_to_rgb_jpeg(x: torch.Tensor) -> torch.Tensor:
    # x: (B,3,H,W), domain Y in [0,255], Cb/Cr centered at 128
    M = torch.tensor([[1., 0., 1.402],
                      [1., -0.344136, -0.714136],
                      [1., 1.772, 0.]], dtype=x.dtype, device=x.device)  # (3,3)
    shift = torch.tensor([0., -128., -128.], dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    M_T = M.t()
    y = torch.einsum('bchw,cd->bdhw', x + shift, M_T)
    return y

# downsampling 4:2:0
def _downsampling_420(x: torch.Tensor):
    # x: (B,3,H,W) -> (y: (B,H,W), cb:(B,H/2,W/2), cr:(B,H/2,W/2))
    y = x[:, 0, :, :]
    cb = x[:, 1:2, :, :]
    cr = x[:, 2:2+1, :, :]
    cb = F.avg_pool2d(cb, kernel_size=2, stride=2)
    cr = F.avg_pool2d(cr, kernel_size=2, stride=2)
    return y, cb.squeeze(1), cr.squeeze(1)

# upsampling 4:2:0
def _upsampling_420(y, cb, cr):
    # y:(B,H,W), cb:(B,H/2,W/2), cr:(B,H/2,W/2) -> (B,3,H,W)
    cb_up = cb.unsqueeze(1).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    cr_up = cr.unsqueeze(1).repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    y = y.unsqueeze(1)
    return torch.cat([y, cb_up, cr_up], dim=1)  # (B,3,H,W)

# image to 8x8 patches
def _image_to_patches(x: torch.Tensor):
    # x: (B,H,W) -> (B, L, 8, 8)
    B, H, W = x.shape
    u = F.unfold(x.unsqueeze(1), kernel_size=8, stride=8)  # (B,64,L)
    L = u.shape[-1]
    u = u.transpose(1, 2).contiguous().view(B, L, 8, 8)
    return u

# 8x8 patches to image
def _patches_to_image(p: torch.Tensor, H: int, W: int):
    # p: (B,L,8,8) -> (B,H,W)
    B, L, _, _ = p.shape
    # .view can fail when tensor is non-contiguous (stride spans multiple subspaces).
    # Use reshape which handles non-contiguous tensors, then transpose and make contiguous
    p = p.reshape(B, L, 64).transpose(1, 2).contiguous()  # (B,64,L)
    out = F.fold(p, output_size=(H, W), kernel_size=8, stride=8)  # (B,1,H,W)
    return out.squeeze(1)

# 8x8 DCT and IDCT
def _dct_mats(dtype, device):
    # Return C (8x8), alpha (8,), scale_outer (8x8)
    C = torch.empty(8, 8, dtype=dtype, device=device)
    for u in range(8):
        for x in range(8):
            C[u, x] = math.cos((2 * x + 1) * u * math.pi / 16.0)
    alpha = torch.tensor([1 / math.sqrt(2)] + [1.0] * 7, dtype=dtype, device=device)
    scale_outer = 0.25 * torch.ger(alpha, alpha)  # (8,8)
    return C, alpha, scale_outer

# 8x8 DCT
def _dct_8x8(blocks: torch.Tensor):
    # blocks: (B,L,8,8) -> (B,L,8,8)
    B, L, _, _ = blocks.shape
    device, dtype = blocks.device, blocks.dtype
    C, alpha, scale_outer = _dct_mats(dtype, device)
    x = blocks - 128.0
    tmp = torch.einsum('ux, blxy -> bluy', C, x)          # sum over x
    out = torch.einsum('vy, bluy -> bluv', C, tmp)        # sum over y
    out = out * scale_outer.view(1, 1, 8, 8)
    return out

# 8x8 IDCT
def _idct_8x8(coefs: torch.Tensor):
    # coefs: (B,L,8,8) -> (B,L,8,8)
    device, dtype = coefs.device, coefs.dtype
    C, alpha, _ = _dct_mats(dtype, device)
    scaled = coefs * torch.ger(alpha, alpha).view(1, 1, 8, 8)  # multiply by alpha_outer
    tmp = torch.einsum('xu, bluv -> blxv', C.t(), scaled)      # sum over u
    out = torch.einsum('yv, blxv -> blxy', C.t(), tmp)         # sum over v
    out = 0.25 * out + 128.0
    return out

# differentiable rounding 
def _round_only_at_0(x: torch.Tensor) -> torch.Tensor:
    cond = (x.abs() < 0.5).to(x.dtype)
    return cond * (x ** 3) + (1.0 - cond) * x

# quantization tables
def _make_tables(dtype, device):
    y_table = torch.tensor(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]], dtype=dtype, device=device).t()  # transpose to match TF
    c_table = torch.empty((8, 8), dtype=dtype, device=device)
    c_table.fill_(99.0)
    c_table[:4, :4] = torch.tensor([[17, 18, 24, 47],
                                    [18, 21, 26, 66],
                                    [24, 26, 56, 99],
                                    [47, 66, 99, 99]], dtype=dtype, device=device).t()
    return y_table, c_table

# differentiable JPEG compress-decompress
def _jpeg_compress_decompress(x: torch.Tensor, factor: torch.Tensor, downsample_c: bool = True):
    # x: (B,3,H,W) in [0,1] -> (B,3,H,W) in [0,1], differentiable
    B, _, H0, W0 = x.shape
    device, dtype = x.device, x.dtype

    x = x * 255.0

    H, W = H0, W0
    if (H % 16) != 0 or (W % 16) != 0:
        H_pad = (16 - (H % 16)) % 16
        W_pad = (16 - (W % 16)) % 16
        # pad bottom and right, symmetric-like (replicate)
        x = F.pad(x, pad=(0, W_pad, 0, H_pad), mode='replicate')
        H, W = x.shape[2], x.shape[3]

    # RGB -> YCbCr (JPEG)
    img = _rgb_to_ycbcr_jpeg(x)  # (B,3,H,W)

    if downsample_c:
        y, cb, cr = _downsampling_420(img)  # y:(B,H,W) cb/cr:(B,H/2,W/2)
    else:
        y = img[:, 0, :, :]
        cb = img[:, 1, :, :]
        cr = img[:, 2, :, :]

    # To patches
    y_p = _image_to_patches(y)             # (B,L,8,8)
    cb_p = _image_to_patches(cb)           # (B,Lc,8,8)
    cr_p = _image_to_patches(cr)           # (B,Lc,8,8)

    # DCT
    y_dct = _dct_8x8(y_p)
    cb_dct = _dct_8x8(cb_p)
    cr_dct = _dct_8x8(cr_p)

    # Quantize (with differentiable rounding)
    y_table, c_table = _make_tables(dtype, device)

    # Broadcast factor (scalar) to tables
    y_q = y_dct / (y_table.view(1, 1, 8, 8) * factor)
    cb_q = cb_dct / (c_table.view(1, 1, 8, 8) * factor)
    cr_q = cr_dct / (c_table.view(1, 1, 8, 8) * factor)

    y_q = _round_only_at_0(y_q)
    cb_q = _round_only_at_0(cb_q)
    cr_q = _round_only_at_0(cr_q)

    # Dequantize
    y_deq = y_q * (y_table.view(1, 1, 8, 8) * factor)
    cb_deq = cb_q * (c_table.view(1, 1, 8, 8) * factor)
    cr_deq = cr_q * (c_table.view(1, 1, 8, 8) * factor)

    # IDCT
    y_rec_p = _idct_8x8(y_deq)
    cb_rec_p = _idct_8x8(cb_deq)
    cr_rec_p = _idct_8x8(cr_deq)

    # Back to images
    y_rec = _patches_to_image(y_rec_p, H, W)
    if downsample_c:
        Hh, Ww = H // 2, W // 2
        cb_rec = _patches_to_image(cb_rec_p, Hh, Ww)
        cr_rec = _patches_to_image(cr_rec_p, Hh, Ww)
        img_rec = _upsampling_420(y_rec, cb_rec, cr_rec)  # (B,3,H,W)
    else:
        cb_rec = _patches_to_image(cb_rec_p, H, W)
        cr_rec = _patches_to_image(cr_rec_p, H, W)
        img_rec = torch.stack([y_rec, cb_rec, cr_rec], dim=1)

    # YCbCr -> RGB
    out = _ycbcr_to_rgb_jpeg(img_rec)

    # Crop to original size
    if (H, W) != (H0, W0):
        out = out[:, :, :H0, :W0]

    out = out.clamp(0.0, 255.0) / 255.0
    return out


class Noise(nn.Module):
    def __init__(self, args, global_step: int, height: int = 400, width: int = 400, kernel_size: int = 7):
        super().__init__()
        self.args = args
        self.global_step = float(global_step)
        self.height = height
        self.width = width
        self.kernel_size = kernel_size

    def forward(self, image_chw: torch.Tensor, args=None, global_step: float = None, *_, **__) -> torch.Tensor:

        self.args = args
        self.global_step = float(global_step)
        device = image_chw.device
        dtype = image_chw.dtype
        B, C, H_in, W_in = image_chw.shape

        # [-1,1] -> [0,1]
        x = ((image_chw + 1.0) / 2.0).clamp(0.0, 1.0)

        # compute ramp
        step = self.global_step
        rnd_bri_amt = _ramp(step, self.args.rnd_bri_ramp) * self.args.rnd_bri
        rnd_hue_amt = _ramp(step, self.args.rnd_hue_ramp) * self.args.rnd_hue
        rnd_noise_cap = _ramp(step, self.args.rnd_noise_ramp) * self.args.rnd_noise
        rnd_sat_cap = _ramp(step, self.args.rnd_sat_ramp) * self.args.rnd_sat

        contrast_low_r = 1.0 - (1.0 - self.args.contrast_low) * _ramp(step, self.args.contrast_ramp)
        contrast_high_r = 1.0 + (self.args.contrast_high - 1.0) * _ramp(step, self.args.contrast_ramp)

        # compute jpeg quality factor
        q_rand = torch.rand(B, device=device, dtype=dtype)
        jpeg_quality = 100.0 - q_rand * _ramp(step, self.args.jpeg_quality_ramp) * (100.0 - self.args.jpeg_quality)
        jpeg_factor = torch.where(jpeg_quality < 50.0,
                                  5000.0 / jpeg_quality,
                                  200.0 - 2.0 * jpeg_quality)
        jpeg_factor = (jpeg_factor / 100.0 + 1e-4).view(B, 1, 1, 1)

        # 1 blur
        blurred = []
        for i in range(B):
            k2d = _random_blur_kernel(self.kernel_size, device=device, dtype=dtype)
            blurred.append(_depthwise_blur_rgb(x[i:i+1], k2d))
        x = torch.cat(blurred, dim=0)

        # 2 additive noise 
        if rnd_noise_cap > 0:
            stds = (torch.rand(B, device=device, dtype=dtype) * float(rnd_noise_cap)).view(B, 1, 1, 1)
            noise = torch.randn_like(x) * stds
            x = (x + noise).clamp(0.0, 1.0)
        else:
            x = x.clamp(0.0, 1.0)

        # 3 color
        # 3.1 contrast
        scales = (torch.rand(B, device=device, dtype=dtype) * (float(contrast_high_r) - float(contrast_low_r)) + float(contrast_low_r)).view(B, 1, 1, 1)
        x = x * scales

        # 3.2 brightness / hue
        hue_vec = torch.empty(B, 3, 1, 1, device=device, dtype=dtype).uniform_(-float(rnd_hue_amt), float(rnd_hue_amt))
        bri_sca = torch.empty(B, 1, 1, 1, device=device, dtype=dtype).uniform_(-float(rnd_bri_amt), float(rnd_bri_amt))
        x = (x + hue_vec + bri_sca).clamp(0.0, 1.0)

        # 3.3 saturation 
        sat = float(rnd_sat_cap) * float(random.random())
        if sat != 0:
            weights = torch.tensor([0.3, 0.6, 0.1], device=device, dtype=dtype).view(1, 3, 1, 1)
            lum = (x * weights).sum(dim=1, keepdim=True)
            x = (1.0 - sat) * x + sat * lum

        if (H_in, W_in) != (self.height, self.width):
            x = F.interpolate(x, size=(self.height, self.width), mode='bilinear', align_corners=False)

        # 4 jpeg
        if not self.args.no_jpeg:
            x = _jpeg_compress_decompress(x, jpeg_factor, downsample_c=True).clamp(0.0, 1.0)
            
        # [0,1] -> [-1,1]
        out = (x * 2.0 - 1.0).clamp(-1.0, 1.0)
        return out
