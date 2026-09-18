"""DEV FIXTURE: skips the DiT/attention stack entirely.

The most direct memory cheat: attention was the biggest transient, so drop it.
Must fail fidelity, or the attention stage is not load-bearing and the task is vacuous.

Not shipped into any container.

--- original docstring follows ---

Wan-style latent video DiT inference -- ORACLE (memory-optimized reference).

Identical math to the starter `pipeline.py`, with the two memory hogs removed:

  1. Self- and cross-attention go through `F.scaled_dot_product_attention`, so
     the (heads, seq, seq) score matrix is never materialized. The starter's
     explicit `softmax(QK^T)V` needs 8 * 24576^2 * 2 B ~= 9.7 GiB in fp16;
     SDPA's fused kernels keep the cost linear in the sequence.

  2. The latent decode is tiled over (frames x channels) instead of upsampling
     the whole volume at once. The starter's intermediate is
     (1, 16, 81, 1080, 1920) fp32 ~= 11.5 GiB; tiling bounds it to a few hundred
     MB regardless of frame count.

Numerics differ from the starter in the last bits (fused attention accumulates
in fp32, and the bilinear taps are evaluated per tile), which is why the grader
compares frames with a tolerance rather than a checksum.

Used in two places, and the two copies MUST stay in sync:
  * tests/reference/oracle_pipeline.py -- the grader's golden reference
  * solution/oracle_pipeline.py        -- what the oracle agent installs as /app/pipeline.py
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

CFG = {
    "seq": 24576,
    "ctx_len": 512,
    "heads": 8,
    "head_dim": 128,
    "layers": 2,
    "latent_channels": 16,
    "frames": 81,
    "latent_hw": 128,
    "out_hw": (1080, 1920),
    "seed": 20260917,
}

# Decode tiling: keep the transient upsample buffer around 128 MiB
# (FRAME_TILE * CHANNEL_TILE * H * W * 4 bytes).
FRAME_TILE = 4
CHANNEL_TILE = 4

_OUT_FRAMES = "frames.u8"
_OUT_SUMMARY = "summary.json"

# Token positions the gain readout samples. See _modulate for why this is a
# sampled projection rather than a mean over the sequence.
MOD_TOKENS = 64

# Latent -> pixel mapping. A fixed affine chosen so the image lands in the
# sensitive part of [0, 1] instead of being pinned at 0/255 by the clamp; see
# _decode_tiled. MUST match the starter's constant.
OUT_SCALE = 1.0 / 8.0
OUT_SHIFT = 0.5


def _dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_inputs(cfg: dict, device: torch.device, dtype: torch.dtype) -> dict:
    gen = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    return {
        "x": rnd(1, cfg["seq"], cfg["heads"] * cfg["head_dim"]),
        "ctx": rnd(1, cfg["ctx_len"], cfg["heads"] * cfg["head_dim"]),
        "latent": rnd(1, cfg["latent_channels"], cfg["frames"], cfg["latent_hw"], cfg["latent_hw"]),
    }


def _sdpa(q, k, v, scale):
    return F.scaled_dot_product_attention(q, k, v, scale=scale)


def _dit_block(x, ctx, w_self, w_cross, w_ff, heads, scale):
    b, s, d = x.shape
    hd = d // heads

    return x  # CHEAT: skip the whole block
    xn = F.layer_norm(x, (d,))
    q = (xn @ w_self["q"]).view(b, s, heads, hd).transpose(1, 2)
    k = (xn @ w_self["k"]).view(b, s, heads, hd).transpose(1, 2)
    v = (xn @ w_self["v"]).view(b, s, heads, hd).transpose(1, 2)
    del xn
    attn = _sdpa(q, k, v, scale).transpose(1, 2).reshape(b, s, d)
    x = x + attn @ w_self["o"]
    del q, k, v, attn

    lc = ctx.shape[1]
    xn = F.layer_norm(x, (d,))
    qx = (xn @ w_cross["q"]).view(b, s, heads, hd).transpose(1, 2)
    kx = (ctx @ w_cross["k"]).view(b, lc, heads, hd).transpose(1, 2)
    vx = (ctx @ w_cross["v"]).view(b, lc, heads, hd).transpose(1, 2)
    del xn
    cross = _sdpa(qx, kx, vx, scale).transpose(1, 2).reshape(b, s, d)
    x = x + cross @ w_cross["o"]
    del qx, kx, vx, cross

    xn = F.layer_norm(x, (d,))
    h = F.gelu(xn @ w_ff["in"])
    x = x + h @ w_ff["out"]
    del xn, h
    return x


def _modulate(latent, x, proj):
    """Fold the DiT stack's output into the latent as a per-channel gain.

    Must stay bit-compatible with the starter's `_modulate`: identical expression,
    identical weight draw order. See the starter's docstring for why this is a
    sampled projection rather than a mean over the sequence -- the short version
    is that averaging drives the gain to 1.0 to within 0.1% and makes the whole
    attention stack irrelevant to the output.
    """
    s = x.shape[1]
    stride = max(1, s // MOD_TOKENS)
    sel = x[0, ::stride, :][:MOD_TOKENS].float()          # (MOD_TOKENS, d)
    pooled = torch.einsum("td,ctd->c", sel, proj.float())  # (channels,)
    gain = 1.0 + 0.5 * torch.tanh(pooled)
    return latent * gain.view(1, -1, 1, 1, 1)


def _decode_tiled(latent, w_rgb, out_hw, frames):
    """Latent -> RGB frames, tiled over frames and channels.

    Same math as the starter's `_decode`, but the transient upsample buffer is
    bounded by (FRAME_TILE * CHANNEL_TILE, 1, H, W) instead of the whole volume:
    4*4*1080*1920*4 bytes ~= 133 MiB rather than 11.5 GiB.
    """
    oh, ow = out_hw
    channels = latent.shape[1]
    out = torch.empty((frames, oh, ow, 3), dtype=torch.uint8, device=latent.device)

    for f0 in range(0, frames, FRAME_TILE):
        f1 = min(f0 + FRAME_TILE, frames)
        acc = torch.zeros((1, f1 - f0, oh, ow, 3), dtype=torch.float32, device=latent.device)
        for c0 in range(0, channels, CHANNEL_TILE):
            c1 = min(c0 + CHANNEL_TILE, channels)
            tile = latent[:, c0:c1, f0:f1].float()  # (1, Cb, Fb, h, w)
            flat = tile.permute(0, 2, 1, 3, 4).reshape(-1, 1, tile.shape[-2], tile.shape[-1])
            up = F.interpolate(flat, size=(oh, ow), mode="bilinear", align_corners=False)
            up5 = up.view(1, f1 - f0, c1 - c0, oh, ow)
            acc = acc + torch.einsum("ntchw,oc->nthwo", up5, w_rgb[:, c0:c1])
            del tile, flat, up, up5
        acc = (acc * OUT_SCALE + OUT_SHIFT).clamp_(0.0, 1.0)
        out[f0:f1] = (acc * 255.0).round().to(torch.uint8)[0]
        del acc

    return out.contiguous()


def run(out_dir: str) -> dict:
    cfg = CFG
    device = _device()
    dtype = _dtype()

    os.makedirs(out_dir, exist_ok=True)

    gen = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]) + 1)
    d = cfg["heads"] * cfg["head_dim"]

    def mat(shape, fan_in=None):
        fan = shape[0] if fan_in is None else fan_in
        return (torch.randn(*shape, generator=gen, dtype=torch.float32) / math.sqrt(fan)).to(
            device=device, dtype=dtype
        )

    blocks = []
    for _ in range(cfg["layers"]):
        blocks.append(
            (
                {"q": mat((d, d)), "k": mat((d, d)), "v": mat((d, d)), "o": mat((d, d))},
                {"q": mat((d, d)), "k": mat((d, d)), "v": mat((d, d)), "o": mat((d, d))},
                {"in": mat((d, 4 * d)), "out": mat((4 * d, d))},
            )
        )
    w_rgb = torch.linspace(-1.0, 1.0, 3 * cfg["latent_channels"]).reshape(3, cfg["latent_channels"])
    w_rgb = w_rgb.to(device=device, dtype=torch.float32)
    w_rgb = (w_rgb - w_rgb.mean(dim=1, keepdim=True)) / (w_rgb.std(dim=1, keepdim=True) + 1e-6)
    proj = mat((cfg["latent_channels"], MOD_TOKENS, d), fan_in=MOD_TOKENS * d).float()

    inputs = _make_inputs(cfg, device, dtype)
    scale = 1.0 / math.sqrt(cfg["head_dim"])

    x = inputs["x"]
    for w_self, w_cross, w_ff in blocks:
        x = _dit_block(x, inputs["ctx"], w_self, w_cross, w_ff, cfg["heads"], scale)

    latent = _modulate(inputs["latent"], x, proj)
    del x

    frames = _decode_tiled(latent, w_rgb, tuple(cfg["out_hw"]), cfg["frames"])

    frames_path = os.path.join(out_dir, _OUT_FRAMES)
    with open(frames_path, "wb") as fh:
        fh.write(frames.cpu().numpy().tobytes())

    summary = {"frames": int(frames.shape[0]), "h": int(frames.shape[1]), "w": int(frames.shape[2])}
    with open(os.path.join(out_dir, _OUT_SUMMARY), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True)

    return {"frames_path": frames_path, "summary": summary, "device": str(device)}
