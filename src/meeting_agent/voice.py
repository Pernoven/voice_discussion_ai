from __future__ import annotations

from meeting_agent.config import AgentConfig
from meeting_agent.events import AgentUtterance


class ConsoleVoice:
    """Development voice that prints agent utterances."""

    def __init__(self, config: AgentConfig) -> None:
        self.agent_name = config.agent_name

    async def speak(self, utterance: AgentUtterance) -> None:
        print(f"[{self.agent_name}] {utterance.text}")
