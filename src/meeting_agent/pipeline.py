from __future__ import annotations

import asyncio
from dataclasses import dataclass

from meeting_agent.asr import ASRBackend
from meeting_agent.brain import AsyncNoteBrain, MeetingBrain
from meeting_agent.ear import TextInputEar
from meeting_agent.events import ListeningSession, TranscriptEvent
from meeting_agent.storage import AegisStore
from meeting_agent.voice import ConsoleVoice


class MeetingAgentPipeline:
    def __init__(
        self,
        ear: TextInputEar,
        brain: MeetingBrain,
        voice: ConsoleVoice,
    ) -> None:
        self.ear = ear
        self.brain = brain
        self.voice = voice

    async def run(self) -> None:
        async for event in self.ear.listen():
            utterance = self.brain.process(event)
            if utterance is None:
                continue
            self.brain.remember_utterance(utterance)
            await self.voice.speak(utterance)


class ClassroomListeningPipeline:
    def __init__(
        self,
        ear: TextInputEar,
        store: AegisStore,
        brain: AsyncNoteBrain,
    ) -> None:
        self.ear = ear
        self.store = store
        self.brain = brain

    async def run(self) -> ListeningSession:
        session = self.store.create_session(
            mode="classroom",
            title="Classroom listening session",
        )
        queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
        worker = asyncio.create_task(self.brain.run(queue))
        try:
            async for event in self.ear.listen():
                stored_event = self.store.insert_transcript_event(session.id, event)
                await queue.put(stored_event)
        finally:
            await queue.put(None)
            await queue.join()
            await worker
        return session


@dataclass(frozen=True)
class TranscribeChunksResult:
    session_id: str
    backend: str
    chunk_count: int
    skipped_chunk_count: int
    transcript_event_count: int


async def transcribe_audio_chunks(
    *,
    store: AegisStore,
    session_id: str,
    backend: ASRBackend,
    language: str | None = None,
    limit: int | None = None,
    speaker: str | None = None,
    brain: AsyncNoteBrain | None = None,
) -> TranscribeChunksResult:
    chunks = store.list_audio_chunks(session_id=session_id)
    pending_chunks = []
    skipped_count = 0
    for chunk in chunks:
        if store.has_asr_transcription(
            audio_chunk_id=chunk.id,
            backend=backend.name,
            language=language,
        ):
            skipped_count += 1
            continue
        if limit is None or len(pending_chunks) < limit:
            pending_chunks.append(chunk)

    note_brain = brain or AsyncNoteBrain(store=store, summary_every=5)
    queue: asyncio.Queue[TranscriptEvent | None] = asyncio.Queue()
    worker = asyncio.create_task(note_brain.run(queue))
    transcript_count = 0

    try:
        for chunk in pending_chunks:
            result = backend.transcribe(chunk, language=language)
            event = TranscriptEvent(
                text=result.text,
                speaker=speaker,
                source=f"asr:{backend.name}",
                start_seconds=chunk.start_seconds,
                end_seconds=chunk.end_seconds,
                audio_chunk_id=chunk.id,
            )
            stored_event = store.insert_transcript_event(session_id, event)
            metadata = {
                **result.metadata,
                "speaker": speaker,
                "audio_source": chunk.source,
                "original_source_path": chunk.original_source_path,
            }
            store.insert_asr_transcription(
                transcript_event_id=stored_event.id,
                audio_chunk_id=chunk.id,
                backend=backend.name,
                language=language,
                metadata=metadata,
            )
            transcript_count += 1
            await queue.put(stored_event)
    finally:
        await queue.put(None)
        await queue.join()
        await worker

    return TranscribeChunksResult(
        session_id=session_id,
        backend=backend.name,
        chunk_count=len(pending_chunks),
        skipped_chunk_count=skipped_count,
        transcript_event_count=transcript_count,
    )
