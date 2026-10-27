"""SIGReg (LeJEPA): keeps embeddings spread like an isotropic Gaussian, so they can't collapse."""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812


def sigreg(z: torch.Tensor, slices: int = 256, generator: torch.Generator | None = None) -> torch.Tensor:
    """Epps-Pulley test statistic of z (N, D) along random 1-D projections, averaged. 0 = perfectly Gaussian.

    Same form as jepa-playground 03-05: compare the empirical characteristic function of each projection
    with the standard normal's, weighted by a Gaussian window.
    """
    if z.ndim != 2 or z.size(0) < 2:
        raise ValueError("z must have shape (N, D) with N >= 2")
    t = torch.linspace(-5, 5, 17, device=z.device, dtype=z.dtype)
    dirs = torch.randn(z.size(1), slices, device=z.device, dtype=z.dtype, generator=generator)
    proj = (z @ F.normalize(dirs, dim=0)).unsqueeze(-1) * t  # (N, slices, len(t))
    window = torch.exp(-(t**2) / 2)
    err = (proj.cos().mean(0) - window) ** 2 + proj.sin().mean(0) ** 2
    return torch.trapezoid(err * window, t, dim=-1).mean() * z.size(0)
