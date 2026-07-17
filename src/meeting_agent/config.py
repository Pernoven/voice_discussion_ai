from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AgentConfig:
    """Runtime knobs for the meeting agent pipeline."""

    agent_name: str = "Aegis"
    summary_every_turns: int = 8
    filler_words: tuple[str, ...] = ("嗯", "呃", "啊", "就是", "那個")


@dataclass(frozen=True)
class RuntimeConfig:
    """Top-level app config."""

    agent: AgentConfig = field(default_factory=AgentConfig)
