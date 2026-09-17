"""Measure the pre-clamp range of the decode stage, to pick an output mapping.

Why this exists: with the naive `clamp(rgb, 0, 1)` output, the pre-clamp values
sit around N(0, 4) -- so ~80% of pixels are pinned at exactly 0 or 255. In that
regime the exported frames stop depending on the latent scale, which is how a
fixture that perturbs the DiT gain produced byte-identical output and passed the
fidelity test. The fix is to choose a fixed affine mapping that keeps the image in
the sensitive part of the range; this script measures what that mapping should be.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ORACLE = os.path.join(HERE, "..", "tasks", "wan-vram-budget", "tests", "reference", "oracle_pipeline.py")


def main() -> int:
    spec = importlib.util.spec_from_file_location("oracle_probe", ORACLE)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["oracle_probe"] = mod
    spec.loader.exec_module(mod)

    captured: dict = {}
    real_modulate = mod._modulate
    real_decode = mod._decode_tiled

    def modulate(latent, x, proj):
        out = real_modulate(latent, x, proj)
        captured["gain"] = None
        # recompute the gain the same way _modulate does, for reporting only
        b, s, d = x.shape
        stride = max(1, s // mod.MOD_TOKENS)
        sel = x[0, ::stride, :][: mod.MOD_TOKENS].float()
        pooled = torch.einsum("td,ctd->c", sel, proj.float())
        captured["pooled"] = pooled.detach().float().cpu()
        captured["gain"] = (1.0 + 0.5 * torch.tanh(pooled)).detach().float().cpu()
        captured["latent_absmean"] = out.float().abs().mean().item()
        return out

    def decode_tiled(latent, w_rgb, out_hw, frames):
        import torch.nn.functional as F

        oh, ow = out_hw
        channels = latent.shape[1]
        samples = []
        for f0 in range(0, min(frames, mod.FRAME_TILE * 2), mod.FRAME_TILE):
            f1 = min(f0 + mod.FRAME_TILE, frames)
            acc = torch.zeros((1, f1 - f0, oh, ow, 3), dtype=torch.float32, device=latent.device)
            for c0 in range(0, channels, mod.CHANNEL_TILE):
                c1 = min(c0 + mod.CHANNEL_TILE, channels)
                tile = latent[:, c0:c1, f0:f1].float()
                flat = tile.permute(0, 2, 1, 3, 4).reshape(-1, 1, tile.shape[-2], tile.shape[-1])
                up = F.interpolate(flat, size=(oh, ow), mode="bilinear", align_corners=False)
                up5 = up.view(1, f1 - f0, c1 - c0, oh, ow)
                acc = acc + torch.einsum("ntchw,oc->nthwo", up5, w_rgb[:, c0:c1])
            samples.append(acc.detach().float().cpu().flatten())
        captured["pre"] = torch.cat(samples)
        # Hand back a real (clamped) result so the rest of run() works.
        return real_decode(latent, w_rgb, out_hw, frames)

    mod._modulate = modulate
    mod._decode_tiled = decode_tiled

    out_dir = tempfile.mkdtemp(prefix="probe_range_")
    import time

    t0 = time.time()
    mod.run(out_dir)
    print(f"ran full-config oracle in {time.time() - t0:.1f}s")

    g = captured["gain"]
    print(f"\ngain           : min={g.min():.3f} max={g.max():.3f} mean={g.mean():.3f} std={g.std():.3f}")
    print(f"pooled         : min={captured['pooled'].min():.2f} max={captured['pooled'].max():.2f}")
    print(f"|latent| mean  : {captured['latent_absmean']:.3f}   (raw latent is ~N(0,1) -> ~0.8)")

    pre = captured["pre"]
    print(f"\npre-clamp rgb  : mean={pre.mean():.3f} std={pre.std():.3f} "
          f"min={pre.min():.2f} max={pre.max():.2f}")
    for lo, hi in ((0.0, 1.0), (-1.0, 2.0), (-2.0, 3.0)):
        inside = ((pre >= lo) & (pre <= hi)).float().mean().item()
        print(f"  P(in [{lo:+.1f},{hi:+.1f}]) = {inside * 100:5.1f}%")
    clip_lo = (pre < 0).float().mean().item()
    clip_hi = (pre > 1).float().mean().item()
    print(f"  clipped low   = {clip_lo * 100:5.1f}%   clipped high = {clip_hi * 100:5.1f}%")

    # Sensitivity: how many uint8 levels does the median pixel move per unit of
    # relative latent scale? Larger is better; near zero means the output is dead.
    for k in (0.9, 1.0, 1.1):
        scaled = (pre * k).clamp(0, 1).mul(255).round()
        base = pre.clamp(0, 1).mul(255).round()
        print(f"  scale x{k:.1f}: changed pixels = {(scaled != base).float().mean() * 100:5.1f}%")

    print("\nSuggested mapping: rgb_out = OUT_SHIFT + OUT_SCALE * rgb_pre, then clamp(0,1)")
    inv = 1.0 / max(pre.std().item() * 4.0, 1e-6)
    print(f"  with OUT_SCALE ~= {inv:.4f} (4 sigma fits in [0,1]) and OUT_SHIFT = 0.5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
