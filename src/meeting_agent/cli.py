from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from meeting_agent.asr import (
    ASRError,
    create_asr_backend,
    format_asr_environment_probe,
    probe_asr_environment,
)
from meeting_agent.audio import (
    AudioInputError,
    chunk_audio_file,
    list_audio_devices,
    record_microphone_chunks,
)
from meeting_agent.benchmark import (
    BenchmarkInputError,
    load_asr_benchmark_manifest,
    run_asr_benchmark,
    write_asr_benchmark_report,
)
from meeting_agent.brain import AsyncNoteBrain
from meeting_agent.brain import MeetingBrain
from meeting_agent.config import RuntimeConfig
from meeting_agent.ear import TextInputEar
from meeting_agent.events import (
    AudioChunk,
    OpenQuestion,
    ResearchNote,
    TranscriptEvent,
)
from meeting_agent.pipeline import (
    ClassroomListeningPipeline,
    MeetingAgentPipeline,
    transcribe_audio_chunks,
)
from meeting_agent.storage import AegisStore, DEFAULT_DB_PATH
from meeting_agent.voice import ConsoleVoice


async def run_legacy_interactive() -> None:
    config = RuntimeConfig()
    pipeline = MeetingAgentPipeline(
        ear=TextInputEar(config.agent),
        brain=MeetingBrain(config.agent),
        voice=ConsoleVoice(config.agent),
    )
    await pipeline.run()


async def run_demo() -> None:
    config = RuntimeConfig()
    brain = MeetingBrain(config.agent)
    voice = ConsoleVoice(config.agent)
    demo_events = (
        TranscriptEvent(text="我們今天要討論 Entropy 和 BCI 的關係", source="demo"),
        TranscriptEvent(text="/summary", source="demo"),
    )
    for event in demo_events:
        utterance = brain.process(event)
        if utterance is None:
            continue
        brain.remember_utterance(utterance)
        await voice.speak(utterance)


async def run_classroom() -> None:
    config = RuntimeConfig()
    store = AegisStore(DEFAULT_DB_PATH)
    try:
        store.initialize()
        brain = AsyncNoteBrain(store=store, summary_every=5)
        pipeline = ClassroomListeningPipeline(
            ear=TextInputEar(config.agent),
            store=store,
            brain=brain,
        )
        print("Classroom listening session started. Enter transcript lines; /quit ends.")
        session = await pipeline.run()
        print(f"Session flushed: {session.id}")
        print_brain_errors(brain)
    finally:
        store.close()


def show_notes() -> None:
    store = AegisStore(DEFAULT_DB_PATH)
    try:
        store.initialize()
        notes = store.list_research_notes()
    finally:
        store.close()
    if not notes:
        print("No research notes found.")
        return
    for note in notes:
        print(format_note(note))


def show_questions() -> None:
    store = AegisStore(DEFAULT_DB_PATH)
    try:
        store.initialize()
        questions = store.list_open_questions()
    finally:
        store.close()
    if not questions:
        print("No open questions found.")
        return
    for question in questions:
        print(format_question(question))


def ingest_audio_chunks(
    path: str,
    chunk_seconds: float,
    max_chunks: int | None = None,
) -> None:
    store = AegisStore(DEFAULT_DB_PATH)
    try:
        store.initialize()
        session = store.create_session(
            mode="audio_file",
            title=f"Audio file ingest: {Path(path).name}",
        )
        chunks = chunk_audio_file(
            source_path=Path(path),
            session_id=session.id,
            chunk_seconds=chunk_seconds,
            max_chunks=max_chunks,
        )
        for chunk in chunks:
            store.insert_audio_chunk(chunk)
    finally:
        store.close()

    print(f"Audio session: {session.id}")
    print(f"Chunks recorded: {len(chunks)}")
    for chunk in chunks:
        print(f"{chunk.path} ({chunk.duration_seconds:.2f}s)")


def simulate_realtime_audio(
    path: str,
    chunk_seconds: float,
    speed: float,
    max_chunks: int | None = None,
) -> None:
    if speed < 0:
        raise AudioInputError("speed must be greater than or equal to 0.")

    store = AegisStore(DEFAULT_DB_PATH)
    try:
        store.initialize()
        session = store.create_session(
            mode="audio_file_simulation",
            title=f"Realtime audio simulation: {Path(path).name}",
        )
        chunks = chunk_audio_file(
            source_path=Path(path),
            session_id=session.id,
            chunk_seconds=chunk_seconds,
            max_chunks=max_chunks,
        )
        print(f"Audio session: {session.id}")
        for chunk in chunks:
            store.insert_audio_chunk(chunk)
            print(format_chunk_event(chunk), flush=True)
            if speed > 0:
                time.sleep(chunk.duration_seconds / speed)
    finally:
        store.close()

    print(f"Chunks recorded: {len(chunks)}")


def print_audio_devices() -> None:
    print(list_audio_devices())


def record_audio_chunks(duration_seconds: float, chunk_seconds: float) -> None:
    store = AegisStore(DEFAULT_DB_PATH)
    try:
        store.initialize()
        session = store.create_session(
            mode="audio_mic",
            title="Microphone audio chunk recording",
        )
        chunks = record_microphone_chunks(
            session_id=session.id,
            duration_seconds=duration_seconds,
            chunk_seconds=chunk_seconds,
        )
        for chunk in chunks:
            store.insert_audio_chunk(chunk)
    finally:
        store.close()

    print(f"Audio session: {session.id}")
    print(f"Chunks recorded: {len(chunks)}")
    for chunk in chunks:
        print(f"{chunk.path} ({chunk.duration_seconds:.2f}s)")


async def run_transcribe_chunks(
    session_id: str,
    backend_name: str,
    language: str | None,
    limit: int | None,
    model_path: str | None,
    speaker: str | None,
) -> None:
    backend = create_asr_backend(backend_name, model_path=model_path)
    store = AegisStore(DEFAULT_DB_PATH)
    try:
        store.initialize()
        brain = AsyncNoteBrain(store=store, summary_every=5)
        result = await transcribe_audio_chunks(
            store=store,
            session_id=session_id,
            backend=backend,
            language=language,
            limit=limit,
            speaker=speaker,
            brain=brain,
        )
        brain_errors = tuple(brain.errors)
    finally:
        store.close()

    print(f"Transcribed session: {result.session_id}")
    print(f"ASR backend: {result.backend}")
    print(f"Chunks processed: {result.chunk_count}")
    print(f"Chunks skipped as duplicates: {result.skipped_chunk_count}")
    print(f"Transcript events written: {result.transcript_event_count}")
    for error in brain_errors:
        print(f"Brain warning: {error}")


def print_brain_errors(brain: AsyncNoteBrain) -> None:
    for error in brain.errors:
        print(f"Brain warning: {error}")


def probe_asr_env(backend_name: str, model_path: str | None) -> None:
    print(
        format_asr_environment_probe(
            probe_asr_environment(backend_name, model_path=model_path)
        )
    )


def benchmark_asr(
    manifest_path: str,
    backend_name: str,
    language: str | None,
    model_path: str | None,
    output_path: str | None,
) -> None:
    backend = create_asr_backend(backend_name, model_path=model_path)
    cases = load_asr_benchmark_manifest(
        Path(manifest_path),
        default_language=language or "zh-CN",
    )
    report = run_asr_benchmark(cases, backend=backend)
    if output_path is None:
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        output = (
            Path(".artifacts/benchmarks")
            / f"{Path(manifest_path).stem}-{backend.name}-{timestamp}.json"
        )
    else:
        output = Path(output_path)
    saved_path = write_asr_benchmark_report(report, output)

    print(f"ASR benchmark backend: {report.backend}")
    print(f"Cases: {report.case_count}")
    print(f"CER: {report.character_error_rate:.4f}")
    print(f"RTF: {report.real_time_factor:.4f}")
    print(f"Audio seconds: {report.total_audio_seconds:.2f}")
    print(f"Inference seconds: {report.total_inference_seconds:.2f}")
    print(f"Report: {saved_path}")


def format_note(note: ResearchNote) -> str:
    return (
        f"[{note.created_at.isoformat()}] {note.note_type}: {note.title}\n"
        f"session={note.session_id} sources={','.join(note.source_transcript_event_ids)}\n"
        f"{note.body}\n"
    )


def format_question(question: OpenQuestion) -> str:
    return (
        f"[{question.created_at.isoformat()}] {question.status}: {question.question}\n"
        f"session={question.session_id} sources={','.join(question.source_transcript_event_ids)}\n"
    )


def format_chunk_event(chunk: AudioChunk) -> str:
    return "chunk_event " + json.dumps(
        {
            "id": chunk.id,
            "path": chunk.path,
            "duration_seconds": chunk.duration_seconds,
            "source": chunk.source,
            "session_id": chunk.session_id,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Aegis meeting agent.")
    parser.add_argument(
        "--classroom",
        action="store_true",
        help="Run continuous classroom listening mode.",
    )
    parser.add_argument(
        "--show-notes",
        action="store_true",
        help="Print research notes from data/aegis.db.",
    )
    parser.add_argument(
        "--show-questions",
        action="store_true",
        help="Print open questions from data/aegis.db.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a deterministic non-interactive demo.",
    )
    parser.add_argument(
        "--legacy-interactive",
        action="store_true",
        help="Run the old console response demo.",
    )
    parser.add_argument(
        "--ingest-audio",
        metavar="PATH",
        help="Ingest an audio file into audio_chunks using the default chunk size.",
    )
    parser.add_argument(
        "--chunk-audio",
        metavar="PATH",
        help="Split an audio file into audio_chunks.",
    )
    parser.add_argument(
        "--simulate-realtime-audio",
        metavar="PATH",
        help="Simulate realtime audio input by creating and printing chunk events.",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=5.0,
        help="Chunk size in seconds for audio file and microphone commands.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Realtime simulation speed. Use 0 for no wait, 10 for 10x speed.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Limit file chunking commands to the first N chunks.",
    )
    parser.add_argument(
        "--list-audio-devices",
        action="store_true",
        help="List microphone input/output devices through sounddevice.",
    )
    parser.add_argument(
        "--record-chunks",
        type=float,
        metavar="SECONDS",
        help="Record microphone audio for SECONDS and save WAV chunks.",
    )
    parser.add_argument(
        "--transcribe-chunks",
        metavar="SESSION_ID",
        help="Transcribe stored audio_chunks for SESSION_ID through an ASR backend.",
    )
    parser.add_argument(
        "--benchmark-asr",
        metavar="MANIFEST",
        help="Run an ASR CER/RTF benchmark from a JSONL manifest.",
    )
    parser.add_argument(
        "--benchmark-output",
        metavar="PATH",
        help="Write --benchmark-asr JSON results to PATH.",
    )
    parser.add_argument(
        "--asr-backend",
        default="fake",
        help="ASR backend name: fake, nemotron-3.5-asr, or whisper.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit --transcribe-chunks to the first N chunks.",
    )
    parser.add_argument(
        "--lang",
        help="Language hint passed to ASR backend metadata, for example zh-CN.",
    )
    parser.add_argument(
        "--speaker",
        help="Speaker label stored with transcripts, for example lecturer.",
    )
    parser.add_argument(
        "--probe-asr-env",
        metavar="BACKEND",
        help="Probe local ASR dependencies without installing models or packages.",
    )
    parser.add_argument(
        "--model-path",
        help=(
            "Local ASR model path for guarded adapters such as nemotron-3.5-asr."
        ),
    )
    args = parser.parse_args()

    try:
        if args.probe_asr_env:
            probe_asr_env(args.probe_asr_env, args.model_path)
        elif args.benchmark_asr:
            benchmark_asr(
                args.benchmark_asr,
                args.asr_backend,
                args.lang,
                args.model_path,
                args.benchmark_output,
            )
        elif args.show_notes:
            show_notes()
        elif args.show_questions:
            show_questions()
        elif args.ingest_audio:
            ingest_audio_chunks(args.ingest_audio, args.chunk_seconds, args.max_chunks)
        elif args.chunk_audio:
            ingest_audio_chunks(args.chunk_audio, args.chunk_seconds, args.max_chunks)
        elif args.simulate_realtime_audio:
            simulate_realtime_audio(
                args.simulate_realtime_audio,
                args.chunk_seconds,
                args.speed,
                args.max_chunks,
            )
        elif args.list_audio_devices:
            print_audio_devices()
        elif args.record_chunks is not None:
            record_audio_chunks(args.record_chunks, args.chunk_seconds)
        elif args.transcribe_chunks:
            asyncio.run(
                run_transcribe_chunks(
                    args.transcribe_chunks,
                    args.asr_backend,
                    args.lang,
                    args.limit,
                    args.model_path,
                    args.speaker,
                )
            )
        elif args.demo:
            asyncio.run(run_demo())
        elif args.legacy_interactive:
            asyncio.run(run_legacy_interactive())
        else:
            asyncio.run(run_classroom())
    except KeyboardInterrupt:
        pass
    except AudioInputError as exc:
        print(f"Audio error: {exc}")
    except ASRError as exc:
        print(f"ASR error: {exc}")
    except BenchmarkInputError as exc:
        print(f"Benchmark error: {exc}")


if __name__ == "__main__":
    main()
