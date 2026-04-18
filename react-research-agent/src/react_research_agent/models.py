from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ResearchStep:
    step: int
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class ResearchNote:
    note_id: int
    claim: str
    evidence: str
    source: str


@dataclass
class ResearchMemory:
    notes: List[ResearchNote] = field(default_factory=list)
    steps: List[ResearchStep] = field(default_factory=list)
    sources: Dict[str, int] = field(default_factory=dict)

    def add_source(self, url: str) -> int:
        if not url:
            return 0
        if url not in self.sources:
            self.sources[url] = len(self.sources) + 1
        return self.sources[url]

    def add_note(self, claim: str, evidence: str, source: str) -> ResearchNote:
        note = ResearchNote(
            note_id=len(self.notes) + 1,
            claim=claim.strip(),
            evidence=evidence.strip(),
            source=source.strip(),
        )
        self.notes.append(note)
        if source.startswith("http"):
            self.add_source(source)
        return note
