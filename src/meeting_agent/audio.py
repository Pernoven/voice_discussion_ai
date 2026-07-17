from __future__ import annotations

import math
import shutil
import subprocess
import wave
from pathlib import Path
from uuid import uuid4

from meeting_agent.events import AudioChunk


DEFAULT_AUDIO_CHUNK_DIR = Path("data/audio_chunks")
SUPPORTED_WAV_SUFFIX = ".wav"
SUPPORTED_COMPRESSED_SUFFIXES = {
    ".aac",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".webm",
    ".wma",
}
ASR_SAMPLE_RATE = 16_000
ASR_CHANNELS = 1


class AudioInputError(RuntimeError):
    """Raised when an audio source cannot be read or recorded."""


class AudioDependencyError(AudioInputError):
    """Raised when optional audio dependencies are not installed."""


def require_wav_path(path: Path) -> None:
    if path.suffix.lower() != SUPPORTED_WAV_SUFFIX:
        raise AudioInputError(
            "目前只支援 WAV 音檔；mp3/m4a 之後會透過 ffmpeg 支援。"
        )


def chunk_wav_file(
    source_path: Path,
    session_id: str,
    output_dir: Path = DEFAULT_AUDIO_CHUNK_DIR,
    chunk_seconds: float = 5.0,
    max_chunks: int | None = None,
) -> list[AudioChunk]:
    """Split a WAV file into chunk WAV files and return metadata rows."""
    source_path = source_path.expanduser()
    require_wav_path(source_path)
    if chunk_seconds <= 0:
        raise AudioInputError("chunk_seconds must be greater than 0.")
    if max_chunks is not None and max_chunks <= 0:
        raise AudioInputError("max_chunks must be greater than 0 when provided.")
    if not source_path.exists():
        raise AudioInputError(f"Audio file not found: {source_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[AudioChunk] = []

    with wave.open(str(source_path), "rb") as reader:
        params = reader.getparams()
        sample_rate = reader.getframerate()
        channels = reader.getnchannels()
        frames_per_chunk = max(1, int(sample_rate * chunk_seconds))
        total_frames = reader.getnframes()
        chunk_count = math.ceil(total_frames / frames_per_chunk) if total_frames else 0

        for index in range(chunk_count):
            if max_chunks is not None and len(chunks) >= max_chunks:
                break
            frames = reader.readframes(frames_per_chunk)
            frame_count = len(frames) // (params.sampwidth * channels)
            if frame_count == 0:
                continue

            chunk_id = str(uuid4())
            chunk_path = output_dir / f"{session_id}_{index:04d}_{chunk_id}.wav"
            with wave.open(str(chunk_path), "wb") as writer:
                writer.setparams(params)
                writer.writeframes(frames)

            chunks.append(
                AudioChunk(
                    id=chunk_id,
                    session_id=session_id,
                    path=str(chunk_path),
                    original_source_path=str(source_path),
                    duration_seconds=frame_count / sample_rate,
                    sample_rate=sample_rate,
                    channels=channels,
                    source="file",
                    start_seconds=(index * frames_per_chunk) / sample_rate,
                    status="recorded",
                )
            )

    return chunks


def chunk_audio_file(
    source_path: Path,
    session_id: str,
    output_dir: Path = DEFAULT_AUDIO_CHUNK_DIR,
    chunk_seconds: float = 5.0,
    max_chunks: int | None = None,
) -> list[AudioChunk]:
    """Split an audio file into WAV chunks ready for later ASR processing."""
    source_path = source_path.expanduser()
    suffix = source_path.suffix.lower()
    if suffix == SUPPORTED_WAV_SUFFIX:
        return chunk_wav_file(
            source_path=source_path,
            session_id=session_id,
            output_dir=output_dir,
            chunk_seconds=chunk_seconds,
            max_chunks=max_chunks,
        )
    if suffix not in SUPPORTED_COMPRESSED_SUFFIXES:
        raise AudioInputError(
            "Unsupported audio file type. Supported formats include WAV, MP3, M4A, "
            "AAC, FLAC, OGG, OPUS, WEBM, MP4, and WMA."
        )
    return chunk_compressed_audio_file(
        source_path=source_path,
        session_id=session_id,
        output_dir=output_dir,
        chunk_seconds=chunk_seconds,
        max_chunks=max_chunks,
    )


def chunk_compressed_audio_file(
    source_path: Path,
    session_id: str,
    output_dir: Path = DEFAULT_AUDIO_CHUNK_DIR,
    chunk_seconds: float = 5.0,
    max_chunks: int | None = None,
) -> list[AudioChunk]:
    """Decode compressed audio through ffmpeg into mono 16 kHz PCM WAV chunks."""
    source_path = source_path.expanduser()
    if chunk_seconds <= 0:
        raise AudioInputError("chunk_seconds must be greater than 0.")
    if max_chunks is not None and max_chunks <= 0:
        raise AudioInputError("max_chunks must be greater than 0 when provided.")
    if not source_path.exists():
        raise AudioInputError(f"Audio file not found: {source_path}")

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise AudioDependencyError(
            "ffmpeg was not found. Install the system ffmpeg CLI to ingest MP3/M4A "
            "or other compressed audio files."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    batch_id = str(uuid4())
    output_pattern = output_dir / f"{session_id}_ffmpeg_{batch_id}_%04d.wav"
    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        str(ASR_CHANNELS),
        "-ar",
        str(ASR_SAMPLE_RATE),
        "-acodec",
        "pcm_s16le",
    ]
    if max_chunks is not None:
        command.extend(["-t", f"{chunk_seconds * max_chunks:.6f}"])
    command.extend(
        [
            "-f",
            "segment",
            "-segment_time",
            f"{chunk_seconds:.6f}",
            "-reset_timestamps",
            "1",
            str(output_pattern),
        ]
    )

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        message = (
            result.stderr.strip()
            or result.stdout.strip()
            or "unknown ffmpeg error"
        )
        raise AudioInputError(f"ffmpeg failed to decode audio: {message}")

    chunk_paths = sorted(output_dir.glob(f"{session_id}_ffmpeg_{batch_id}_*.wav"))
    if max_chunks is not None:
        chunk_paths = chunk_paths[:max_chunks]

    chunks: list[AudioChunk] = []
    start_seconds = 0.0
    for chunk_path in chunk_paths:
        metadata = read_wav_metadata(chunk_path)
        duration_seconds = float(metadata["duration_seconds"])
        chunks.append(
            AudioChunk(
                id=str(uuid4()),
                session_id=session_id,
                path=str(chunk_path),
                original_source_path=str(source_path),
                duration_seconds=duration_seconds,
                sample_rate=metadata["sample_rate"],
                channels=metadata["channels"],
                source="file",
                start_seconds=start_seconds,
                status="recorded",
            )
        )
        start_seconds += duration_seconds

    return chunks


def read_wav_metadata(path: Path) -> dict[str, float | int]:
    with wave.open(str(path), "rb") as reader:
        sample_rate = reader.getframerate()
        frame_count = reader.getnframes()
        return {
            "duration_seconds": frame_count / sample_rate if sample_rate else 0.0,
            "sample_rate": sample_rate,
            "channels": reader.getnchannels(),
        }


def list_audio_devices() -> str:
    sounddevice = _import_sounddevice()
    return str(sounddevice.query_devices())


def record_microphone_chunks(
    session_id: str,
    duration_seconds: float,
    output_dir: Path = DEFAULT_AUDIO_CHUNK_DIR,
    chunk_seconds: float = 5.0,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> list[AudioChunk]:
    if duration_seconds <= 0:
        raise AudioInputError("duration_seconds must be greater than 0.")
    if chunk_seconds <= 0:
        raise AudioInputError("chunk_seconds must be greater than 0.")

    sounddevice = _import_sounddevice()
    numpy = _import_numpy()

    output_dir.mkdir(parents=True, exist_ok=True)
    total_frames = int(duration_seconds * sample_rate)
    recording = sounddevice.rec(
        total_frames,
        samplerate=sample_rate,
        channels=channels,
        dtype="int16",
    )
    sounddevice.wait()

    frames_per_chunk = max(1, int(chunk_seconds * sample_rate))
    chunks: list[AudioChunk] = []
    for index, start in enumerate(range(0, len(recording), frames_per_chunk)):
        chunk_data = recording[start : start + frames_per_chunk]
        if len(chunk_data) == 0:
            continue

        chunk_id = str(uuid4())
        chunk_path = output_dir / f"{session_id}_{index:04d}_{chunk_id}.wav"
        with wave.open(str(chunk_path), "wb") as writer:
            writer.setnchannels(channels)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(numpy.asarray(chunk_data, dtype=numpy.int16).tobytes())

        chunks.append(
            AudioChunk(
                id=chunk_id,
                session_id=session_id,
                path=str(chunk_path),
                original_source_path=None,
                duration_seconds=len(chunk_data) / sample_rate,
                sample_rate=sample_rate,
                channels=channels,
                source="mic",
                start_seconds=start / sample_rate,
                status="recorded",
            )
        )

    return chunks


def _import_sounddevice():
    try:
        import sounddevice
    except ImportError as exc:
        raise AudioDependencyError(
            "Audio device support requires optional dependency 'sounddevice'. "
            "Install the audio extra before using microphone commands."
        ) from exc
    return sounddevice


def _import_numpy():
    try:
        import numpy
    except ImportError as exc:
        raise AudioDependencyError(
            "Microphone recording requires optional dependency 'numpy'. "
            "Install the audio extra before using microphone commands."
        ) from exc
    return numpy
