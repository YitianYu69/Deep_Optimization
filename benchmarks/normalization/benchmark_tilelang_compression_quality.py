"""CPU-only screen of existing LayerNorm saved-state compression settings.

This measures codec approximation quality, not TileLang CUDA correctness or
performance. No setting is selected automatically and no runtime default changes.
"""

import argparse
import json
from pathlib import Path


def relative_error(actual, expected):
    delta = actual.double() - expected.double()
    return {"max_abs": delta.abs().max().item(),
            "relative_l2": (delta.norm() / expected.double().norm().clamp_min(1e-30)).item()}


def screen(*, rows=32, columns=768, bits=(2, 4, 8), groups=(128, 256, 512), seed=42,
           distributions=("normal",), eps=1e-5):
    import torch
    from deepopt.opt.quantization.activation import quantize_pack_activation, dequantize_unpack_activation

    if rows < 1 or columns < 1 or eps <= 0:
        raise ValueError("rows, columns and eps must be positive")
    if not bits or any(value not in (1, 2, 4, 8) for value in bits) or not groups or min(groups) < 1:
        raise ValueError("use supported bit widths and positive group sizes")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    records = []
    for distribution in distributions:
        x = torch.randn(rows, columns, generator=generator)
        if distribution == "large_offset":
            x = 10000 + x * .01
        elif distribution == "outliers":
            x[:, ::31] *= 32
        elif distribution != "normal":
            raise ValueError(f"Unknown distribution {distribution}")
        variance, mean = torch.var_mean(x.double(), dim=1, correction=0)
        mean = mean.float()
        inverse = torch.rsqrt(variance.float() + eps)
        h = (x - mean[:, None]) * inverse[:, None]
        dy = torch.randn(rows, columns, generator=generator)
        gamma = torch.randn(columns, generator=generator)
        u = dy * gamma

        def gradients(saved):
            return ((u - u.mean(1, keepdim=True) - saved * (u * saved).mean(1, keepdim=True)) * inverse[:, None],
                    (dy * saved).sum(0), dy.sum(0))

        reference = gradients(h)
        for width in dict.fromkeys(bits):
            for requested_group in dict.fromkeys(groups):
                group = min(requested_group, columns)
                packed = quantize_pack_activation(h, group, (1, group), bits=width)
                decoded = dequantize_unpack_activation(packed, dtype=torch.float32)
                shift = decoded.mean(1)
                centered = decoded - shift[:, None]
                variance_sum = centered.square().sum(1)
                target = h.square().sum(1)
                gain = torch.where(variance_sum > 0,
                    torch.sqrt(target / variance_sum.clamp_min(1e-12 * columns)), 0.)
                corrected = centered * gain[:, None]
                actual = gradients(corrected)
                state = sum(tensor.numel() * tensor.element_size() for tensor in
                            (packed.data, packed.step, packed.minimum, shift, gain))
                errors = {name: relative_error(value, expected) for name, value, expected in
                          zip(("dx", "dweight", "dbias"), actual, reference)}
                records.append({"distribution": distribution, "bits": width,
                    "requested_group_size": requested_group, "realized_group_size": group,
                    "saved_xhat_and_correction_bytes": state,
                    "uncompressed_xhat_bytes": h.numel() * h.element_size(),
                    "xhat": relative_error(corrected, h), "gradient_errors_vs_uncompressed": errors,
                    "mean_max_abs_error": corrected.mean(1).abs().max().item(),
                    "second_moment_max_abs_error": (corrected.square().mean(1) - h.square().mean(1)).abs().max().item()})
    return {"schema_version": 1, "execution": "cpu_codec_quality", "seed": seed,
            "shape": [rows, columns], "eps": eps,
            "notes": ["Independent CPU decode and effective LayerNorm mean/second-moment correction.",
                      "Errors compare the saved-state backward rule against uncompressed saved values.",
                      "Not CUDA/backend parity, an acceptance threshold, or training convergence evidence.",
                      "Saved-byte comparison covers xhat and correction only, not invariant mean/rstd/model state.",
                      "Existing settings are screened without changing Triton or TileLang defaults."],
            "results": records}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--columns", type=int, default=768)
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--group-sizes", type=int, nargs="+", default=[1024])
    parser.add_argument("--distributions", nargs="+", choices=("normal", "large_offset", "outliers"), default=["normal"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    import torch
    torch.set_num_threads(args.threads)
    try:
        report = screen(rows=args.rows, columns=args.columns, bits=args.bits, groups=args.group_sizes,
                        distributions=args.distributions, seed=args.seed)
    except ValueError as error:
        parser.error(str(error))
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
