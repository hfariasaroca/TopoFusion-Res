#!/usr/bin/env python3
"""Author-code, random-tensor smoke test. No climate data or checkpoint loading."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
sys.path.insert(0, str(MODELS))

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is needed for the synthetic test; see docs/ENVIRONMENT.md") from exc

from topofusion_base import ModelECMIPSwinUNetTopography
from model_e2_residual_head import ModelE2ResidualHead


def main():
    torch.manual_seed(20260813)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    batch = {
        "dynamic": torch.randn(1, 24, 22, 22),
        "coarse_static": torch.randn(1, 1, 22, 22),
        "fine_static": torch.randn(1, 6, 64, 64),
        "time_features": torch.randn(1, 3),
        "ghg_features": torch.randn(1, 4),
    }
    base = ModelECMIPSwinUNetTopography().eval()
    residual = ModelE2ResidualHead().eval()
    with torch.no_grad():
        pred = base(**batch)
        result = residual(**batch, return_parts=True)
    assert tuple(pred.shape) == (1, 1, 64, 64), pred.shape
    assert tuple(result["tas_final"].shape) == (1, 1, 64, 64)
    assert torch.isfinite(pred).all() and torch.isfinite(result["tas_final"]).all()
    assert float(result["residual"].abs().max()) < 1e-7, "Residual head should initialize at zero"
    print("PASS: Model E + E2 synthetic forward shapes, finite outputs and zero residual initialization")
    print("NOTE: This does NOT test weights, climate data, training, SSP3-7.0 or paper metrics.")


if __name__ == "__main__":
    main()
