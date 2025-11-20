from __future__ import annotations

from typing import List, Literal, Optional

from covertreex.api import Runtime as ApiRuntime
from profiles.loader import ProfileError
from profiles.overrides import OverrideError


def resolve_metric_flag(
    metric: Literal["auto", "euclidean", "residual", "residual-lite"],
    *,
    profile: Optional[str],
    overrides: Optional[List[str]],
) -> Literal["euclidean", "residual", "residual-lite"]:
    """Return the effective metric derived from CLI inputs."""

    if metric in ("euclidean", "residual", "residual-lite"):
        return metric
    if not profile:
        return "euclidean"
    try:
        runtime = ApiRuntime.from_profile(profile, overrides=overrides)
    except (ProfileError, OverrideError, ValueError) as exc:
        raise ValueError(f"Invalid profile or overrides: {exc}") from exc
    described_metric = (runtime.describe().get("metric") or "euclidean").lower()
    if "residual_correlation_lite" in described_metric or "residual-lite" in described_metric:
        return "residual-lite"
    return "residual" if "residual" in described_metric else "euclidean"


__all__ = ["resolve_metric_flag"]
