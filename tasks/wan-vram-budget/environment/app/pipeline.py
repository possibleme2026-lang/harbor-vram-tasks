"""Wan-style latent video DiT inference -- STARTER (correct but memory-hungry).

This is the file you must optimize. It is functionally correct: it produces the
reference frames for the configured problem size. It is also far too greedy for
the target device and will fail the grader's VRAM gate unchanged.

Contract (do not change the interface):

  CFG  -- module-level dict. The grader reads it and asserts that the problem
          size is unchanged. You may NOT shrink seq / frames / out_hw / layers.
  run(out_dir) -> dict
          Runs inference and writes exactly two files:
            <out_dir>/frames.u8    uint8, C-contiguous, shape (frames, H, W, 3)
            <out_dir>/summary.json JSON object with at least
                                   {"frames": int, "h": int, "w": int}
          Returns a dict of diagnostics (free-form; the grader ignores it).

The grader re-runs this file in a separate container under a hard 8 GiB device
memory budget. See /app/vram_probe.py to reproduce that locally.
"""

from __future__ import annotations

import json
import math
import os

import torch
import torch.nn.functional as F

CFG = {
    # Self-attention sequence length = latent video tokens.
    "seq": 24576,
    # Text-context length for cross-attention.
    "ctx_len": 512,
    "heads": 8,
    "head_dim": 128,
    "layers": 2,
    "latent_channels": 16,
    "frames": 81,
    "latent_hw": 128,
    # Frames are returned at this resolution.
    "out_hw": (1080, 1920),
    "seed": 20260917,
}

_OUT_FRAMES = "frames.u8"
_OUT_SUMMARY = "summary.json"

# Token positions the gain readout samples. See _modulate for why this is a
# sampled projection rather than a mean over the sequence.
MOD_TOKENS = 64

# Latent -> pixel mapping. A fixed affine chosen so the image lands in the
# sensitive part of [0, 1] instead of being pinned at 0/255 by the clamp; see
# _decode.
OUT_SCALE = 1.0 / 8.0
OUT_SHIFT = 0.5


def _dtype() -> torch.dtype:
    return torch.float16 if torch.cuda.is_available() else torch.float32


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_inputs(cfg: dict, device: torch.device, dtype: torch.dtype) -> dict:
    """Deterministic synthetic inputs -- no checkpoint needed, fully reproducible."""
    gen = torch.Generator(device="cpu").manual_seed(int(cfg["seed"]))

    def rnd(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float32).to(device=device, dtype=dtype)

    return {
        "x": rnd(1, cfg["seq"], cfg["heads"] * cfg["head_dim"]),
        "ctx": rnd(1, cfg["ctx_len"], cfg["heads"] * cfg["head_dim"]),
        "latent": rnd(1, cfg["latent_channels"], cfg["frames"], cfg["latent_hw"], cfg["latent_hw"]),
    }


def _naive_sdpa(q, k, v, scale):
    """Textbook attention. Materializes the full (heads, S, S) score matrix.

    For seq=24576, heads=8 in fp16 that is 8 * 24576^2 * 2 bytes ~= 9.7 GiB,
    which alone exceeds the target budget.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def _dit_block(x, ctx, w_self, w_cross, w_ff, heads, scale):
    b, s, d = x.shape
    hd = d // heads

    # ---- self attention ----
    xn = F.layer_norm(x, (d,))
    q = (xn @ w_self["q"]).view(b, s, heads, hd).transpose(1, 2)
    k = (xn @ w_self["k"]).view(b, s, heads, hd).transpose(1, 2)
    v = (xn @ w_self["v"]).view(b, s, heads, hd).transpose(1, 2)
    del xn
    attn = _naive_sdpa(q, k, v, scale)
    attn = attn.transpose(1, 2).reshape(b, s, d)
    x = x + attn @ w_self["o"]
    del q, k, v, attn

    # ---- cross attention onto the text context ----
    lc = ctx.shape[1]
    xn = F.layer_norm(x, (d,))
    qx = (xn @ w_cross["q"]).view(b, s, heads, hd).transpose(1, 2)
    kx = (ctx @ w_cross["k"]).view(b, lc, heads, hd).transpose(1, 2)
    vx = (ctx @ w_cross["v"]).view(b, lc, heads, hd).transpose(1, 2)
    del xn
    cross = _naive_sdpa(qx, kx, vx, scale)
    cross = cross.transpose(1, 2).reshape(b, s, d)
    x = x + cross @ w_cross["o"]
    del qx, kx, vx, cross

    # ---- feed forward ----
    xn = F.layer_norm(x, (d,))
    h = F.gelu(xn @ w_ff["in"])
    x = x + h @ w_ff["out"]
    del xn, h
    return x


def _modulate(latent, x, proj):
    """Fold the DiT stack's output into the latent as a per-channel gain.

    This is what makes the attention stage load-bearing for the exported frames:
    the grader compares your frames against a reference, so an implementation
    that skips or approximates the blocks produces a different gain and therefore
    different pixels.

    Note the readout is a *projection of sampled token positions*, not a mean over
    the sequence. Averaging is the trap here: `x` has seq=24576 tokens drawn from
    roughly i.i.d. distributions, so a mean has standard deviation ~1/sqrt(seq) ~
    0.006, the gain collapses to 1.0 to within 0.1%, and the entire attention
    stack stops affecting the output -- an implementation that deleted it
    entirely would then pass the fidelity check. The projection keeps the gain
    O(1) while still depending on every token through attention.
    """
    s = x.shape[1]
    stride = max(1, s // MOD_TOKENS)
    sel = x[0, ::stride, :][:MOD_TOKENS].float()          # (MOD_TOKENS, d)
    pooled = torch.einsum("td,ctd->c", sel, proj.float())  # (channels,)
    gain = 1.0 + 0.5 * torch.tanh(pooled)
    return latent * gain.view(1, -1, 1, 1, 1)


def _decode(latent, w_rgb, out_hw):
    """Latent -> RGB frames. Upsamples the whole volume at once.

    `F.interpolate` in bilinear mode wants 4D, so the frame and channel axes are
    folded into the batch dimension. The upsampled result is
    (81*16, 1, 1080, 1920) fp32 ~= 11.5 GiB, and it is held as a single buffer
    (the 5D view is free) -- which is what makes this stage exceed the budget.
    """
    oh, ow = out_hw
    n, c, t, h, w = latent.shape
    flat = latent.permute(0, 2, 1, 3, 4).reshape(n * t * c, 1, h, w)
    up = F.interpolate(flat.float(), size=(oh, ow), mode="bilinear", align_corners=False)
    up5 = up.view(n, t, c, oh, ow)
    rgb = torch.einsum("ntchw,oc->nthwo", up5, w_rgb)
    rgb = (rgb * OUT_SCALE + OUT_SHIFT).clamp_(0.0, 1.0)
    frames = (rgb * 255.0).round().to(torch.uint8)
    return frames[0].contiguous()  # (T, H, W, 3)


def run(out_dir: str) -> dict:
    cfg = CFG
    device = _device()
    dtype = _dtype()

    os.makedirs(out_dir, exist_ok=True)

    # Fixed weights: drawn once from the seed so the pipeline needs no checkpoint.
    # Scaled by 1/sqrt(fan_in) so the residual stream stays O(1) -- unscaled randn
    # weights make the activations grow layer over layer and overflow fp16.
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
    # Gain-readout projection, scaled so the einsum in _modulate is O(1). Drawn at
    # this exact point in the sequence so the starter and the reference generate
    # bit-identical weights.
    proj = mat((cfg["latent_channels"], MOD_TOKENS, d), fan_in=MOD_TOKENS * d).float()

    inputs = _make_inputs(cfg, device, dtype)
    scale = 1.0 / math.sqrt(cfg["head_dim"])

    x = inputs["x"]
    for w_self, w_cross, w_ff in blocks:
        x = _dit_block(x, inputs["ctx"], w_self, w_cross, w_ff, cfg["heads"], scale)

    latent = _modulate(inputs["latent"], x, proj)
    del x

    frames = _decode(latent, w_rgb, tuple(cfg["out_hw"]))

    frames_path = os.path.join(out_dir, _OUT_FRAMES)
    with open(frames_path, "wb") as fh:
        fh.write(frames.cpu().numpy().tobytes())

    summary = {"frames": int(frames.shape[0]), "h": int(frames.shape[1]), "w": int(frames.shape[2])}
    with open(os.path.join(out_dir, _OUT_SUMMARY), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, sort_keys=True)

    return {"frames_path": frames_path, "summary": summary, "device": str(device)}
