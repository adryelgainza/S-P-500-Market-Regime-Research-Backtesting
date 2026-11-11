"""Explicit, validated inputs for the supported research and notebook entry points."""

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RunConfig:
    mode: str = "replay"
    dataset_id: str = "sha256-f670b664d7872145a0ef05e6a6aefd40cd62a54d1bff2052a452bd185eb8a55f"
    protocol: str = "protocols/layer6_validation_v2.json"
    start: str = "2014-01-02"
    end: str = "2026-07-24"
    seed: int = 42
    numerical_rtol: float = 1e-10
    numerical_atol: float = 1e-12
    cell_timeout_seconds: int = 1800

    def __post_init__(self):
        if self.mode not in {"replay", "fit"}:
            raise ValueError("Research mode must be replay or fit")
        if not self.dataset_id.startswith("sha256-") or len(self.dataset_id) != 71:
            raise ValueError("Use an exact content-addressed dataset ID")
        if pd.Timestamp(self.start) >= pd.Timestamp(self.end):
            raise ValueError("The research start must precede its exclusive end")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Use a nonnegative integer seed")
        if (not 0 < self.numerical_rtol <= 1e-8 or not 0 < self.numerical_atol <= 1e-10
                or not isinstance(self.cell_timeout_seconds, int) or self.cell_timeout_seconds < 1):
            raise ValueError("Invalid numerical tolerance or cell timeout")
        path = Path(self.protocol)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Protocol must be a project-relative path")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_json(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(value)-{field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown run settings {sorted(unknown)}")
        return cls(**value)
