import asyncio
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_agent.brain import AsyncNoteBrain
from meeting_agent.config import AgentConfig
from meeting_agent.ear import TextInputEar
from meeting_agent.events import TranscriptEvent
from meeting_agent.llm import LLMAnalysis, LLMAnalysisError, NoteDraft
from meeting_agent.pipeline import ClassroomListeningPipeline
from meeting_agent.storage import AegisStore


class StaticEar(TextInputEar):
    def __init__(self, texts: tuple[str, ...]) -> None:
        super().__init__(AgentConfig())
        self.texts = texts

    async def listen(self):
        for text in self.texts:
            yield TranscriptEvent(text=text, source="test")


class DynamicAnalyzer:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []

    def analyze(self, events, *, create_summary):
        self.thread_ids.append(threading.get_ident())
        current_text = events[-1].text
        concepts = ()
        questions = ()
        if "湧現" in current_text:
            concepts = (
                NoteDraft(
                    title="湧現行為",
                    body="系統由局部互動形成整體模式的現象。",
                ),
            )
        if "尚未解釋" in current_text:
            questions = ("局部互動如何形成可預測的整體模式？",)
        summary = (
            NoteDraft(title="課堂重點", body="課堂正在討論複雜系統的湧現行為。")
            if create_summary
            else None
        )
        return LLMAnalysis(
            summary=summary,
            concepts=concepts,
            questions=questions,
        )


class FailingAnalyzer:
    def analyze(self, events, *, create_summary):
        raise LLMAnalysisError("local model unavailable")


class ClassroomPipelineTest(unittest.TestCase):
    def test_pipeline_flushes_transcripts_notes_and_questions(self) -> None:
        texts = (
            "今天介紹複雜系統",
            "局部單元會彼此互動",
            "這種整體模式稱為湧現",
            "形成機制目前尚未解釋",
            "最後整理實驗觀察",
        )
        analyzer = DynamicAnalyzer()
        main_thread_id = threading.get_ident()
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "aegis.db"
            store = AegisStore(db_path)
            try:
                store.initialize()
                pipeline = ClassroomListeningPipeline(
                    ear=StaticEar(texts),
                    store=store,
                    brain=AsyncNoteBrain(
                        store,
                        analyzer=analyzer,
                        summary_every=5,
                    ),
                )

                asyncio.run(pipeline.run())

                transcript_count = store.connection.execute(
                    "SELECT COUNT(*) FROM transcript_events"
                ).fetchone()[0]
                notes = store.list_research_notes()
                questions = store.list_open_questions()
            finally:
                store.close()

        self.assertEqual(transcript_count, 5)
        self.assertTrue(any(note.note_type == "summary" for note in notes))
        self.assertTrue(any(note.note_type == "concept" for note in notes))
        self.assertGreaterEqual(len(questions), 1)
        self.assertTrue(analyzer.thread_ids)
        self.assertTrue(
            all(thread_id != main_thread_id for thread_id in analyzer.thread_ids)
        )

    def test_llm_failure_preserves_transcript_and_finishes_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = AegisStore(Path(tmpdir) / "aegis.db")
            try:
                store.initialize()
                brain = AsyncNoteBrain(store, analyzer=FailingAnalyzer())
                pipeline = ClassroomListeningPipeline(
                    ear=StaticEar(("這段原始逐字稿必須保存",)),
                    store=store,
                    brain=brain,
                )

                session = asyncio.run(pipeline.run())
                transcripts = store.list_transcript_events(session_id=session.id)
            finally:
                store.close()

        self.assertEqual([event.text for event in transcripts], ["這段原始逐字稿必須保存"])
        self.assertEqual(len(brain.errors), 1)


if __name__ == "__main__":
    unittest.main()
