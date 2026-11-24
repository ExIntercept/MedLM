"""Taint and provenance metadata for external content."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EvidenceSource(str, Enum):
    SYSTEM = "system"
    USER = "user"
    RETRIEVED = "retrieved"
    IMAGE = "image_derived"
    MODEL_OUTPUT = "model_output"


@dataclass
class ProvenanceSegment:
    text: str
    source: EvidenceSource
    meta: dict = field(default_factory=dict)
    quarantined: list[str] = field(default_factory=list)

    @property
    def may_instruct(self) -> bool:
        return self.source in {EvidenceSource.SYSTEM, EvidenceSource.USER}

    @property
    def tainted(self) -> bool:
        return not self.may_instruct
