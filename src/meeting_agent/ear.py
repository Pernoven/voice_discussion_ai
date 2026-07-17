from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator

from meeting_agent.config import AgentConfig
from meeting_agent.events import TranscriptEvent


class TextPreprocessor:
    def __init__(self, filler_words: tuple[str, ...]) -> None:
        self.filler_words = filler_words

    def clean(self, text: str) -> str:
        cleaned = text.strip()
        for filler in self.filler_words:
            cleaned = cleaned.replace(filler, "")
        return " ".join(cleaned.split())


class TextInputEar:
    """Development ear that treats stdin lines as transcript events."""

    def __init__(self, config: AgentConfig) -> None:
        self.preprocessor = TextPreprocessor(config.filler_words)

    async def listen(self) -> AsyncIterator[TranscriptEvent]:
        loop = asyncio.get_running_loop()
        while True:
            if sys.stdin.isatty():
                print("> ", end="", flush=True)
            line = await self._read_stdin_line()
            if line == "":
                return
            text = self.preprocessor.clean(line)
            if not text:
                continue
            if text == "/quit":
                return
            yield TranscriptEvent(text=text, source="stdin")

    async def _read_stdin_line(self) -> str:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[str] = loop.create_future()
        stdin_fd = sys.stdin.fileno()

        def read_ready() -> None:
            if not future.done():
                future.set_result(sys.stdin.readline())

        loop.add_reader(stdin_fd, read_ready)
        try:
            return await future
        finally:
            loop.remove_reader(stdin_fd)
