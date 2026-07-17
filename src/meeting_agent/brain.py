from __future__ import annotations

import asyncio
from meeting_agent.config import AgentConfig
from meeting_agent.events import (
    AgentUtterance,
    MeetingState,
    OpenQuestion,
    ResearchNote,
    TranscriptEvent,
)
from meeting_agent.llm import (
    LLMAnalysis,
    LLMAnalysisError,
    LLMInferenceWorker,
    LocalGemmaAnalyzer,
    TranscriptAnalyzer,
)
from meeting_agent.storage import AegisStore
class MeetingBrain:
    """Legacy summary-only brain kept for the old console demo."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.state = MeetingState()

    def process(self, event: TranscriptEvent) -> AgentUtterance | None:
        if event.text == "/summary":
            return self._summary("manual_summary")

        self.state.append_transcript(event)

        if self._should_periodically_summarize():
            return self._summary("periodic_summary")

        return None

    def remember_utterance(self, utterance: AgentUtterance) -> None:
        self.state.append_utterance(utterance)

    def _should_periodically_summarize(self) -> bool:
        every = self.config.summary_every_turns
        if every <= 0:
            return False
        return len(self.state.transcript) % every == 0

    def _summary(self, reason: str) -> AgentUtterance:
        recent = self.state.recent_text()
        if not recent:
            text = "目前還沒有足夠討論內容可以整理。"
        else:
            text = f"目前最近幾段討論重點是：{recent}"
        return AgentUtterance(text=text, reason=reason, priority=1)


class AsyncNoteBrain:
    """Background LLM note brain that keeps inference off the event loop."""

    def __init__(
        self,
        store: AegisStore,
        analyzer: TranscriptAnalyzer | None = None,
        summary_every: int = 5,
        context_window: int = 8,
    ) -> None:
        if context_window <= 0:
            raise ValueError("context_window must be greater than 0")
        self.store = store
        self.analyzer = analyzer or LocalGemmaAnalyzer.from_environment()
        self.inference_worker = LLMInferenceWorker(self.analyzer)
        self.summary_every = summary_every
        self.context_window = context_window
        self._events: list[TranscriptEvent] = []
        self._noted_concepts: set[str] = set()
        self._asked_questions: set[str] = set()
        self.errors: list[str] = []

    async def run(
        self,
        queue: asyncio.Queue[TranscriptEvent | None],
    ) -> None:
        try:
            while True:
                event = await queue.get()
                try:
                    if event is None:
                        return
                    try:
                        await self.process(event)
                    except Exception as exc:
                        self.errors.append(
                            f"event={event.id}: unexpected brain error: "
                            f"{type(exc).__name__}: {exc}"
                        )
                finally:
                    queue.task_done()
        finally:
            self.inference_worker.close()

    async def process(self, event: TranscriptEvent) -> None:
        if event.session_id is None:
            raise ValueError("TranscriptEvent must have session_id before brain processing")

        self._events.append(event)
        create_summary = (
            self.summary_every > 0
            and len(self._events) % self.summary_every == 0
        )
        recent_events = tuple(self._events[-self.context_window :])
        try:
            analysis = await self.inference_worker.analyze(
                recent_events,
                create_summary=create_summary,
            )
        except LLMAnalysisError as exc:
            self.errors.append(f"event={event.id}: {exc}")
            return

        self._persist_analysis(
            event,
            recent_events,
            analysis,
            create_summary=create_summary,
        )

    def _persist_analysis(
        self,
        event: TranscriptEvent,
        recent_events: tuple[TranscriptEvent, ...],
        analysis: LLMAnalysis,
        *,
        create_summary: bool,
    ) -> None:
        session_id = event.session_id or ""
        if create_summary and analysis.summary is not None:
            self.store.insert_research_note(
                ResearchNote(
                    session_id=session_id,
                    source_transcript_event_ids=tuple(
                        item.id for item in recent_events
                    ),
                    note_type="summary",
                    title=analysis.summary.title,
                    body=analysis.summary.body,
                )
            )

        for concept in analysis.concepts:
            concept_key = concept.title.casefold()
            if concept_key in self._noted_concepts:
                continue
            self._noted_concepts.add(concept_key)
            self.store.insert_research_note(
                ResearchNote(
                    session_id=session_id,
                    source_transcript_event_ids=(event.id,),
                    note_type="concept",
                    title=concept.title,
                    body=concept.body,
                )
            )

        for question in analysis.questions:
            question_key = question.casefold()
            if question_key in self._asked_questions:
                continue
            self._asked_questions.add(question_key)
            self.store.insert_open_question(
                OpenQuestion(
                    session_id=session_id,
                    source_transcript_event_ids=(event.id,),
                    question=question,
                )
            )
