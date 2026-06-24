"""Shared types: digest dataclasses, Candidate/MetaColsResult, Pydantic schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pydantic import BaseModel

from .llm_client import LLMUnavailable  # re-exported; defined in llm_client


@dataclass
class ColumnProfile:
    name: str
    dtype: str            # "categorical" | "string" | "integer" | "float" | "bool"
    n_unique: int
    n_missing: int
    example_values: list
    cells_per_group: dict  # {"min": int, "max": int, "median": float}
    balance: float         # min_group / max_group, 0..1
    unique_per_cell: bool
    single_value: bool
    looks_like_barcode: bool
    is_numeric: bool = False
    v_min: float = 0.0
    v_max: float = 0.0
    v_median: float = 0.0
    v_mean: float = 0.0
    frac_nonneg: float = 0.0
    frac_unit: float = 0.0          # fraction of non-missing values in [0,1]
    is_integer_valued: bool = False


@dataclass
class CompositeProfile:
    columns: list
    n_unique: int
    cells_per_group: dict
    balance: float

    @property
    def label(self) -> str:
        return " + ".join(self.columns)


@dataclass
class BarcodeProfile:
    delimiter: str
    position: str          # "prefix" | "suffix"
    n_groups: int
    cells_per_group: dict
    balance: float
    example_groups: list

    @property
    def label(self) -> str:
        return f"<barcode:{self.position}:{self.delimiter}>"


@dataclass
class ObsDigest:
    n_obs: int
    columns: list                 # list[ColumnProfile]
    composite_candidates: list    # list[CompositeProfile]
    barcode: BarcodeProfile | None = None

    def to_prompt_dict(self) -> dict:
        return {
            "n_obs": self.n_obs,
            "columns": [vars(c) for c in self.columns],
            "composite_candidates": [
                {"columns": c.columns, "n_unique": c.n_unique,
                 "cells_per_group": c.cells_per_group, "balance": c.balance}
                for c in self.composite_candidates
            ],
            "barcode": (
                {"delimiter": self.barcode.delimiter, "position": self.barcode.position,
                 "n_groups": self.barcode.n_groups,
                 "cells_per_group": self.barcode.cells_per_group,
                 "balance": self.barcode.balance,
                 "example_groups": self.barcode.example_groups}
                if self.barcode is not None else None
            ),
        }


@dataclass
class Candidate:
    role: str
    column: str
    kind: str              # "single" | "composite" | "barcode"
    score: float
    reason: str
    source: str            # "llm" | "heuristic"


@dataclass
class MetaColsResult:
    roles: dict            # role_key -> list[Candidate], sorted desc, truncated
    method: str
    digest: ObsDigest

    def top(self, role: str):
        cands = self.roles.get(role) or []
        return cands[0] if cands else None


class RankedCandidate(BaseModel):
    role: str
    column: str
    kind: str = "single"
    score: float
    reason: str


class RankedCandidates(BaseModel):
    candidates: List[RankedCandidate]


class Adjudication(BaseModel):
    role: str
    column: str
    reason: str


class Adjudications(BaseModel):
    verdicts: List[Adjudication]
