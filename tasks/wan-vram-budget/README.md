# terminal-bench/wan-vram-budget

Make a correct-but-greedy latent video DiT inference pipeline fit an 8 GiB device
memory budget without changing the frames it produces.

Category `ML` / subcategory `Inference`. Expert time estimate: ~12 h.

## Difficulty explanation

Two independent memory hogs have to be found and fixed, and the obvious fix for
one of them is not enough: the self-attention path materializes a `(heads, seq,
seq)` score matrix that is quadratic in the latent token count (~9.7 GiB at
`seq=24576`), and the decode path upsamples the entire latent volume to 1080p in
fp32 as a single tensor (~11.5 GiB). Both exceed the budget on their own, so a
partial fix still fails — verified, not assumed: fixing only the attention stage
still peaks at 10.35 GiB. Neither hog is visible as a bug in the code; both read
as ordinary, idiomatic PyTorch, which is why an agent has to actually measure
rather than reason from the source. On top of that, the cheap version of the
attention fix changes the numerics, so the agent must also reason about how much
error its optimization is allowed to introduce.

## Solution explanation

Route both attention paths through fused kernels (`F.scaled_dot_product_attention`)
so the score matrix is never materialized, and tile the decode over frames and
channels so the transient upsample buffer is bounded by the tile instead of the
whole volume. The single most important insight is that "fits in memory" is a
property of the *peak* of the largest transient, not of the total data flow: the
pipeline's total allocation is well above the budget, but no single stage needs
to hold more than a few hundred MB at once. The reference implementation lives in
`tests/reference/oracle_pipeline.py` and `solution/oracle_pipeline.py`.

## Verification explanation

The verifier runs in a separate GPU container with the PyTorch caching allocator
capped at 8 GiB (`set_per_process_memory_fraction`) and the device-memory
reporting APIs patched to agree with the cap, so the pipeline actually OOMs when
it exceeds the budget rather than merely being reported as over. Peak memory is
then measured out-of-process via NVML, which the pipeline cannot influence, and
the emitted frames are compared against a reference run at the same problem size
(mean absolute difference <= 2, max <= 12, PSNR >= 40 dB on 0-255 values). A
separate test asserts `CFG` is unchanged, so the problem cannot be made to fit by
shrinking it. The gates were validated against four arms, not just the happy
path: the reference passes 8/8; the unmodified starter fails on memory (OOM at
9.38 GiB); and two adversarial arms that *do* fit the budget — one perturbing the
attention-dependent readout, one deleting the attention stack entirely — are both
rejected by the fidelity test alone (mean|diff| 10.99 and 13.76 against a 2.0
tolerance). A grader self-check additionally asserts the reference frames are
non-degenerate, so a flat reference cannot silently make every comparison pass.

## Relevant experience

Built and profiled the underlying optimization work on a single 8 GB consumer
laptop GPU (RTX 5060 Laptop, sm_120) for the Wan2.1 video diffusion family, which
is where both of these failure patterns — a fused-attention path silently not
being taken, and a full-volume fp32 decode intermediate — showed up as OOMs and
multi-second stalls in practice.
