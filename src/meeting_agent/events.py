from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ListeningSession:
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)
    mode: str = "classroom"
    title: str | None = None


@dataclass(frozen=True)
class TranscriptEvent:
    text: str
    id: str = field(default_factory=lambda: str(uuid4()))
    session_id: str | None = None
    speaker: str | None = None
    source: str = "unknown"
    start_seconds: float | None = None
    end_seconds: float | None = None
    audio_chunk_id: str | None = None
    created_at: datetime = field(default_factory=utc_now)

    @property
    def timestamp(self) -> datetime:
        return self.created_at


@dataclass(frozen=True)
class ResearchNote:
    session_id: str
    source_transcript_event_ids: tuple[str, ...]
    note_type: str
    title: str
    body: str
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class OpenQuestion:
    session_id: str
    source_transcript_event_ids: tuple[str, ...]
    question: str
    status: str = "open"
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class AudioChunk:
    session_id: str
    path: str
    original_source_path: str | None
    duration_seconds: float
    sample_rate: int
    channels: int
    source: str
    start_seconds: float = 0.0
    status: str = "recorded"
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=utc_now)

    @property
    def end_seconds(self) -> float:
        return self.start_seconds + self.duration_seconds


@dataclass(frozen=True)
class AgentUtterance:
    text: str
    reason: str
    priority: int = 1
    timestamp: datetime = field(default_factory=utc_now)


@dataclass
class MeetingState:
    transcript: list[TranscriptEvent] = field(default_factory=list)
    utterances: list[AgentUtterance] = field(default_factory=list)

    def append_transcript(self, event: TranscriptEvent) -> None:
        self.transcript.append(event)

    def append_utterance(self, utterance: AgentUtterance) -> None:
        self.utterances.append(utterance)

    def recent_text(self, limit: int = 5) -> str:
        events = self.transcript[-limit:]
        return "\n".join(event.text for event in events)
