from __future__ import annotations

import json
import time
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from meeting_agent.asr import ASRBackend
from meeting_agent.audio import read_wav_metadata
from meeting_agent.events import AudioChunk


class BenchmarkInputError(ValueError):
    """Raised when an ASR benchmark manifest is invalid."""


@dataclass(frozen=True)
class ASRBenchmarkCase:
    id: str
    audio_path: Path
    reference: str
    language: str
    speaker: str | None = None


@dataclass(frozen=True)
class ASRBenchmarkCaseResult:
    id: str
    audio_path: str
    reference: str
    hypothesis: str
    language: str
    detected_language: str | None
    speaker: str | None
    reference_characters: int
    edit_distance: int
    character_error_rate: float
    audio_duration_seconds: float
    inference_seconds: float
    real_time_factor: float


@dataclass(frozen=True)
class ASRBenchmarkReport:
    created_at: str
    backend: str
    case_count: int
    reference_characters: int
    edit_distance: int
    character_error_rate: float
    total_audio_seconds: float
    total_inference_seconds: float
    real_time_factor: float
    cases: tuple[ASRBenchmarkCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_cer_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        character
        for character in normalized
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )


def character_error_rate(reference: str, hypothesis: str) -> float:
    errors, reference_length = _character_error_counts(reference, hypothesis)
    if reference_length == 0:
        raise BenchmarkInputError(
            "CER reference is empty after whitespace and punctuation normalization."
        )
    return errors / reference_length


def load_asr_benchmark_manifest(
    manifest_path: Path,
    *,
    default_language: str = "zh-CN",
) -> list[ASRBenchmarkCase]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise BenchmarkInputError(f"Benchmark manifest not found: {manifest_path}")

    cases: list[ASRBenchmarkCase] = []
    for line_number, raw_line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            entry = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise BenchmarkInputError(
                f"Invalid JSON on benchmark manifest line {line_number}: {exc}"
            ) from exc
        if not isinstance(entry, dict):
            raise BenchmarkInputError(
                f"Benchmark manifest line {line_number} must be a JSON object."
            )

        audio_value = entry.get("audio_filepath") or entry.get("audio_path")
        reference = entry.get("reference")
        if not isinstance(audio_value, str) or not audio_value.strip():
            raise BenchmarkInputError(
                f"Benchmark manifest line {line_number} requires audio_filepath."
            )
        if not isinstance(reference, str) or not normalize_cer_text(reference):
            raise BenchmarkInputError(
                f"Benchmark manifest line {line_number} requires a non-empty reference."
            )

        audio_path = Path(audio_value).expanduser()
        if not audio_path.is_absolute():
            audio_path = manifest_path.parent / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise BenchmarkInputError(
                f"Benchmark audio not found on line {line_number}: {audio_path}"
            )

        case_id = entry.get("id") or f"case-{line_number:04d}"
        language = entry.get("language") or default_language
        speaker = entry.get("speaker")
        if not isinstance(case_id, str) or not case_id.strip():
            raise BenchmarkInputError(
                f"Benchmark manifest line {line_number} has an invalid id."
            )
        if not isinstance(language, str) or not language.strip():
            raise BenchmarkInputError(
                f"Benchmark manifest line {line_number} has an invalid language."
            )
        if speaker is not None and not isinstance(speaker, str):
            raise BenchmarkInputError(
                f"Benchmark manifest line {line_number} has an invalid speaker."
            )

        cases.append(
            ASRBenchmarkCase(
                id=case_id,
                audio_path=audio_path,
                reference=reference,
                language=language,
                speaker=speaker,
            )
        )

    if not cases:
        raise BenchmarkInputError("Benchmark manifest contains no cases.")
    return cases


def run_asr_benchmark(
    cases: list[ASRBenchmarkCase],
    *,
    backend: ASRBackend,
) -> ASRBenchmarkReport:
    if not cases:
        raise BenchmarkInputError("At least one benchmark case is required.")
    backend.validate_runtime()

    case_results: list[ASRBenchmarkCaseResult] = []
    total_errors = 0
    total_reference_characters = 0
    total_audio_seconds = 0.0
    total_inference_seconds = 0.0

    for case in cases:
        audio_metadata = read_wav_metadata(case.audio_path)
        duration_seconds = float(audio_metadata["duration_seconds"])
        chunk = AudioChunk(
            id=case.id,
            session_id="asr-benchmark",
            path=str(case.audio_path),
            original_source_path=str(case.audio_path),
            duration_seconds=duration_seconds,
            sample_rate=int(audio_metadata["sample_rate"]),
            channels=int(audio_metadata["channels"]),
            source="benchmark",
        )

        started_at = time.perf_counter()
        transcription = backend.transcribe(chunk, language=case.language)
        inference_seconds = time.perf_counter() - started_at
        errors, reference_characters = _character_error_counts(
            case.reference,
            transcription.text,
        )
        if reference_characters == 0:
            raise BenchmarkInputError(
                f"Benchmark case {case.id} has an empty normalized reference."
            )
        real_time_factor = (
            inference_seconds / duration_seconds if duration_seconds > 0 else 0.0
        )
        detected_language = transcription.metadata.get("detected_language")

        case_results.append(
            ASRBenchmarkCaseResult(
                id=case.id,
                audio_path=str(case.audio_path),
                reference=case.reference,
                hypothesis=transcription.text,
                language=case.language,
                detected_language=(
                    str(detected_language) if detected_language is not None else None
                ),
                speaker=case.speaker,
                reference_characters=reference_characters,
                edit_distance=errors,
                character_error_rate=errors / reference_characters,
                audio_duration_seconds=duration_seconds,
                inference_seconds=inference_seconds,
                real_time_factor=real_time_factor,
            )
        )
        total_errors += errors
        total_reference_characters += reference_characters
        total_audio_seconds += duration_seconds
        total_inference_seconds += inference_seconds

    return ASRBenchmarkReport(
        created_at=datetime.now(timezone.utc).isoformat(),
        backend=backend.name,
        case_count=len(case_results),
        reference_characters=total_reference_characters,
        edit_distance=total_errors,
        character_error_rate=total_errors / total_reference_characters,
        total_audio_seconds=total_audio_seconds,
        total_inference_seconds=total_inference_seconds,
        real_time_factor=(
            total_inference_seconds / total_audio_seconds
            if total_audio_seconds > 0
            else 0.0
        ),
        cases=tuple(case_results),
    )


def write_asr_benchmark_report(
    report: ASRBenchmarkReport,
    output_path: Path,
) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return output_path


def _character_error_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    normalized_reference = normalize_cer_text(reference)
    normalized_hypothesis = normalize_cer_text(hypothesis)
    return (
        _levenshtein_distance(normalized_reference, normalized_hypothesis),
        len(normalized_reference),
    )


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]
