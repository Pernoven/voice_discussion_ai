import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_agent.asr import (
    ASRBackendUnavailableError,
    FakeASRBackend,
    NemotronASRBackend,
    create_asr_backend,
    probe_asr_environment,
)
from meeting_agent.brain import AsyncNoteBrain
from meeting_agent.events import AudioChunk
from meeting_agent.llm import LLMAnalysis, NoteDraft
from meeting_agent.pipeline import transcribe_audio_chunks
from meeting_agent.storage import AegisStore


class SummaryAnalyzer:
    def analyze(self, events, *, create_summary):
        return LLMAnalysis(
            summary=(
                NoteDraft(title="ASR 摘要", body="兩段測試轉錄已完成。")
                if create_summary
                else None
            ),
            concepts=(),
            questions=(),
        )


class ASRPipelineTest(unittest.TestCase):
    def test_fake_backend_transcribes_chunks_and_flushes_brain(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            store = AegisStore(tmp_path / "aegis.db")
            try:
                store.initialize()
                session = store.create_session(mode="audio_file", title="fixture")
                for index in range(2):
                    chunk_path = tmp_path / f"chunk-{index}.wav"
                    chunk_path.write_bytes(b"fake wav bytes")
                    store.insert_audio_chunk(
                        AudioChunk(
                            id=f"chunk-{index:04d}",
                            session_id=session.id,
                            path=str(chunk_path),
                            original_source_path=str(tmp_path / "source.mp3"),
                            duration_seconds=1.0,
                            sample_rate=16000,
                            channels=1,
                            source="file",
                            start_seconds=float(index),
                        )
                    )

                result = asyncio.run(
                    transcribe_audio_chunks(
                        store=store,
                        session_id=session.id,
                        backend=FakeASRBackend(),
                        language="zh-CN",
                        limit=2,
                        speaker="lecturer",
                        brain=AsyncNoteBrain(
                            store,
                            analyzer=SummaryAnalyzer(),
                            summary_every=2,
                        ),
                    )
                )

                rerun_result = asyncio.run(
                    transcribe_audio_chunks(
                        store=store,
                        session_id=session.id,
                        backend=FakeASRBackend(),
                        language="zh-CN",
                        limit=2,
                        speaker="lecturer",
                    )
                )

                transcripts = store.list_transcript_events(session_id=session.id)
                notes = store.list_research_notes()
                asr_rows = store.connection.execute(
                    """
                    SELECT backend, language, metadata_json
                    FROM asr_transcriptions
                    ORDER BY created_at ASC
                    """
                ).fetchall()
            finally:
                store.close()

        self.assertEqual(result.chunk_count, 2)
        self.assertEqual(result.skipped_chunk_count, 0)
        self.assertEqual(result.transcript_event_count, 2)
        self.assertEqual(rerun_result.chunk_count, 0)
        self.assertEqual(rerun_result.skipped_chunk_count, 2)
        self.assertEqual(rerun_result.transcript_event_count, 0)
        self.assertEqual(len(transcripts), 2)
        self.assertTrue(all(event.source == "asr:fake" for event in transcripts))
        self.assertIn("Transcribed chunk chunk", transcripts[0].text)
        self.assertTrue(all(event.speaker == "lecturer" for event in transcripts))
        self.assertEqual(transcripts[0].start_seconds, 0.0)
        self.assertEqual(transcripts[0].end_seconds, 1.0)
        self.assertEqual(transcripts[0].audio_chunk_id, "chunk-0000")
        self.assertTrue(any(note.note_type == "summary" for note in notes))
        self.assertEqual(len(asr_rows), 2)
        self.assertTrue(all(row["backend"] == "fake" for row in asr_rows))
        self.assertTrue(all(row["language"] == "zh-CN" for row in asr_rows))

    def test_nemotron_backend_reports_clear_missing_nemo_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "nemotron.nemo"
            model_path.write_bytes(b"fake nemo")
            backend = create_asr_backend("nemotron-3.5-asr", model_path=model_path)

            self.assertIsInstance(backend, NemotronASRBackend)

            chunk = AudioChunk(
                id="chunk-0001",
                session_id="session-0001",
                path=str(Path(tmpdir) / "chunk.wav"),
                original_source_path=None,
                duration_seconds=1.0,
                sample_rate=16000,
                channels=1,
                source="file",
                start_seconds=10.0,
            )
            with patch(
                "importlib.import_module",
                side_effect=ModuleNotFoundError("No module named 'nemo'"),
            ):
                with self.assertRaisesRegex(
                    ASRBackendUnavailableError,
                    "nemo.collections.asr",
                ):
                    backend.transcribe(chunk, language="en-US")

    def test_nemotron_backend_transcribes_with_language_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            model_path = tmp_path / "nemotron.nemo"
            model_path.write_bytes(b"fake nemo")
            chunk_path = tmp_path / "chunk.wav"
            chunk_path.write_bytes(b"fake wav")
            chunk = AudioChunk(
                id="chunk-0001",
                session_id="session-0001",
                path=str(chunk_path),
                original_source_path=None,
                duration_seconds=5.0,
                sample_rate=16000,
                channels=1,
                source="file",
                start_seconds=10.0,
            )
            backend = NemotronASRBackend(model_path=model_path)
            captured_manifest: dict[str, object] = {}

            def fake_transcribe(
                manifest_paths: list[str],
                *,
                target_lang: str,
                timestamps: bool,
                verbose: bool,
            ) -> list[SimpleNamespace]:
                nonlocal captured_manifest
                captured_manifest = json.loads(
                    Path(manifest_paths[0]).read_text(encoding="utf-8")
                )
                self.assertEqual(target_lang, "en-US")
                self.assertTrue(timestamps)
                self.assertFalse(verbose)
                return [
                    SimpleNamespace(
                        text="Test transcript. <en-US>",
                        timestamp={
                            "word": [
                                {
                                    "word": "Test",
                                    "start": 0.2,
                                    "end": 0.4,
                                },
                                {
                                    "word": "<en-US>",
                                    "start": 0.4,
                                    "end": 0.5,
                                },
                            ],
                            "segment": [
                                {
                                    "segment": "Test transcript.",
                                    "start": 0.2,
                                    "end": 0.8,
                                }
                            ],
                        },
                    )
                ]

            fake_model = Mock()
            fake_model.transcribe.side_effect = fake_transcribe
            backend._model = fake_model
            backend._runtime_metadata = {"device": "cuda"}

            with patch.object(backend, "validate_runtime"):
                result = backend.transcribe(chunk, language="en-US")

        self.assertEqual(result.text, "Test transcript.")
        self.assertEqual(result.metadata["detected_language"], "en-US")
        self.assertEqual(
            result.metadata["word_timestamps"],
            [
                {
                    "text": "Test",
                    "start_seconds": 10.2,
                    "end_seconds": 10.4,
                }
            ],
        )
        self.assertEqual(
            result.metadata["segment_timestamps"][0]["start_seconds"],
            10.2,
        )
        self.assertEqual(captured_manifest["lang"], "en-US")
        self.assertEqual(
            captured_manifest["audio_filepath"],
            str(chunk_path.resolve()),
        )

    def test_reserved_whisper_backend_reports_clear_unavailable_error(self) -> None:
        with self.assertRaisesRegex(ASRBackendUnavailableError, "reserved"):
            create_asr_backend("whisper")

    def test_nemotron_probe_does_not_crash(self) -> None:
        probe = probe_asr_environment("nemotron-3.5-asr")

        self.assertEqual(probe.backend, "nemotron-3.5-asr")
        self.assertIn(
            probe.readiness,
            {
                "ready",
                "model present, missing nemo",
                "missing model",
                "missing cuda",
                "missing dependencies",
            },
        )

    def test_nemotron_probe_reports_local_model_path_and_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "fake.nemo"
            model_path.write_bytes(b"fake nemo bytes")

            probe = probe_asr_environment(
                "nemotron-3.5-asr",
                model_path=model_path,
            )

        self.assertEqual(probe.local_model_path, str(model_path))
        self.assertTrue(probe.local_model_present)
        self.assertEqual(probe.local_model_size_bytes, len(b"fake nemo bytes"))


if __name__ == "__main__":
    unittest.main()
