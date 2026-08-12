"""Check local feature helper compatibility with ``exordium``.

The script compares normalization and padding behavior on deterministic
synthetic tensors. No FI samples are read.
"""

from __future__ import annotations

import torch

from fi import pad_or_crop_time_dim, standardization
from exordium.utils.normalize import standardization as original_standardization
from exordium.utils.padding import pad_or_crop_time_dim as original_pad_or_crop


def main() -> None:
    torch.manual_seed(42)

    mean = torch.randn(25)
    std = torch.rand(25) + 0.1
    x = torch.randn(37, 25)
    ours = standardization(x, mean, std)
    reference = original_standardization(x, mean=mean, std=std)
    torch.testing.assert_close(ours, reference, rtol=0.0, atol=0.0)

    for length, target in [(37, 64), (64, 64), (91, 64)]:
        sequence = torch.randn(length, 25)
        ours_x, ours_mask = pad_or_crop_time_dim(sequence, target)
        ref_x, ref_mask = original_pad_or_crop(sequence, target)
        torch.testing.assert_close(ours_x, ref_x, rtol=0.0, atol=0.0)
        if not torch.equal(ours_mask, ref_mask):
            raise AssertionError(
                f"Mask mismatch for length={length}, target={target}: "
                f"ours={ours_mask.dtype}/{ours_mask.shape}, "
                f"reference={ref_mask.dtype}/{ref_mask.shape}"
            )

    print("PASS: fi.py helpers are bitwise compatible with exordium helpers")


if __name__ == "__main__":
    main()
