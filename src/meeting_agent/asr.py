from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from meeting_agent.events import AudioChunk


FAKE_BACKEND_NAME = "fake"
NEMOTRON_BACKEND_NAME = "nemotron-3.5-asr"
WHISPER_BACKEND_NAME = "whisper"
DEFAULT_NEMOTRON_MODEL_PATH = Path(
    "nemotron/nemotron-3.5-asr-streaming-0.6b.nemo"
)
NEMOTRON_DEFAULT_LANGUAGE = "auto"
_NEMOTRON_LANGUAGE_TAG = re.compile(
    r"\s*<(?P<language>[A-Za-z]{2,3}(?:-[A-Za-z0-9]+)?)>\s*$"
)
SUPPORTED_ASR_BACKENDS = (
    FAKE_BACKEND_NAME,
    NEMOTRON_BACKEND_NAME,
    WHISPER_BACKEND_NAME,
)


class ASRError(RuntimeError):
    """Raised when an ASR backend cannot transcribe a chunk."""


class ASRBackendUnavailableError(ASRError):
    """Raised when a reserved ASR backend is not executable in this milestone."""


@dataclass(frozen=True)
class ASRResult:
    text: str
    metadata: dict[str, Any]


class ASRBackend(ABC):
    name: str

    def validate_runtime(self) -> None:
        """Validate backend runtime before batch work starts."""

    @abstractmethod
    def transcribe(
        self,
        chunk: AudioChunk,
        *,
        language: str | None = None,
    ) -> ASRResult:
        """Transcribe one recorded audio chunk into text plus backend metadata."""


class FakeASRBackend(ASRBackend):
    name = FAKE_BACKEND_NAME

    def transcribe(
        self,
        chunk: AudioChunk,
        *,
        language: str | None = None,
    ) -> ASRResult:
        short_id = chunk.id.split("-", maxsplit=1)[0][:8]
        filename = Path(chunk.path).name
        return ASRResult(
            text=f"Transcribed chunk {short_id} from {filename}",
            metadata={
                "backend": self.name,
                "audio_chunk_id": chunk.id,
                "language": language,
                "duration_seconds": chunk.duration_seconds,
                "sample_rate": chunk.sample_rate,
                "channels": chunk.channels,
                "source_path": chunk.path,
                "chunk_start_seconds": chunk.start_seconds,
                "chunk_end_seconds": chunk.end_seconds,
            },
        )


class NemotronASRBackend(ASRBackend):
    name = NEMOTRON_BACKEND_NAME

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = _resolve_nemotron_model_path(model_path)
        self._model: Any | None = None
        self._runtime_metadata: dict[str, Any] = {}

    def transcribe(
        self,
        chunk: AudioChunk,
        *,
        language: str | None = None,
    ) -> ASRResult:
        self.validate_runtime()
        audio_path = Path(chunk.path).expanduser().resolve()
        if not audio_path.is_file():
            raise ASRError(
                "ASR backend 'nemotron-3.5-asr' found no audio chunk at "
                f"{audio_path}."
            )

        target_language = language or NEMOTRON_DEFAULT_LANGUAGE
        manifest_path = self._write_manifest(
            audio_path=audio_path,
            duration_seconds=chunk.duration_seconds,
            language=target_language,
        )
        try:
            model = self._load_model()
            outputs = model.transcribe(
                [str(manifest_path)],
                target_lang=target_language,
                timestamps=True,
                verbose=False,
            )
            if not outputs:
                raise ASRError(
                    "Nemotron returned no transcription result for "
                    f"audio chunk {chunk.id}."
                )
            text, detected_language = _normalize_nemotron_output(outputs[0])
            timestamp_metadata = _normalize_nemotron_timestamps(
                outputs[0],
                chunk_start_seconds=chunk.start_seconds,
            )
        except ASRError:
            raise
        except Exception as exc:
            raise ASRError(
                "Nemotron transcription failed for audio chunk "
                f"{chunk.id}: {type(exc).__name__}: {exc}"
            ) from exc
        finally:
            manifest_path.unlink(missing_ok=True)

        return ASRResult(
            text=text,
            metadata={
                "backend": self.name,
                "audio_chunk_id": chunk.id,
                "language": target_language,
                "detected_language": detected_language,
                "duration_seconds": chunk.duration_seconds,
                "sample_rate": chunk.sample_rate,
                "channels": chunk.channels,
                "source_path": chunk.path,
                "chunk_start_seconds": chunk.start_seconds,
                "chunk_end_seconds": chunk.end_seconds,
                "model_path": str(self.model_path),
                **timestamp_metadata,
                **self._runtime_metadata,
            },
        )

    def validate_runtime(self) -> None:
        if not self.model_path.exists():
            raise ASRBackendUnavailableError(
                "ASR backend 'nemotron-3.5-asr' found no local .nemo model at "
                f"{self.model_path}. Pass --model-path or download the model before "
                "running Nemotron inference."
            )
        try:
            importlib.import_module("nemo.collections.asr")
        except Exception as exc:
            raise ASRBackendUnavailableError(
                "ASR backend 'nemotron-3.5-asr' requires NVIDIA NeMo ASR runtime "
                "(nemo.collections.asr), but it is not importable: "
                f"{type(exc).__name__}: {exc}. Install and verify NeMo before "
                "running local Nemotron inference."
            ) from exc

        try:
            torch = importlib.import_module("torch")
        except Exception as exc:
            raise ASRBackendUnavailableError(
                "ASR backend 'nemotron-3.5-asr' requires PyTorch, but it is not "
                f"importable: {type(exc).__name__}: {exc}."
            ) from exc
        if not torch.cuda.is_available():
            raise ASRBackendUnavailableError(
                "ASR backend 'nemotron-3.5-asr' requires a CUDA-visible NVIDIA "
                "GPU in the current Python process."
            )

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        nemo = importlib.import_module("nemo")
        nemo_asr = importlib.import_module("nemo.collections.asr")
        torch = importlib.import_module("torch")
        previous_weights_setting = os.environ.get(
            "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"
        )
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            model = nemo_asr.models.ASRModel.restore_from(
                str(self.model_path.resolve()),
                map_location="cuda",
            )
        finally:
            if previous_weights_setting is None:
                os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = (
                    previous_weights_setting
                )

        model.eval()
        self._model = model
        self._runtime_metadata = {
            "device": "cuda",
            "nemo_version": getattr(nemo, "__version__", None),
            "torch_version": getattr(torch, "__version__", None),
        }
        return model

    @staticmethod
    def _write_manifest(
        *,
        audio_path: Path,
        duration_seconds: float,
        language: str,
    ) -> Path:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as manifest:
            json.dump(
                {
                    "audio_filepath": str(audio_path),
                    "duration": duration_seconds,
                    "text": "",
                    "lang": language,
                },
                manifest,
            )
            manifest.write("\n")
            return Path(manifest.name)


def create_asr_backend(name: str, model_path: str | Path | None = None) -> ASRBackend:
    normalized = name.strip().casefold()
    if normalized == FAKE_BACKEND_NAME:
        return FakeASRBackend()
    if normalized == NEMOTRON_BACKEND_NAME:
        return NemotronASRBackend(model_path=model_path)
    if normalized == WHISPER_BACKEND_NAME:
        raise ASRBackendUnavailableError(
            "ASR backend 'whisper' is reserved as a fallback/baseline, but the "
            "adapter and dependencies are not installed/implemented in this "
            "milestone."
        )
    raise ASRBackendUnavailableError(
        "Unknown ASR backend "
        f"'{name}'. Supported backend names: {', '.join(SUPPORTED_ASR_BACKENDS)}."
    )


@dataclass(frozen=True)
class ASREnvironmentProbe:
    backend: str
    python_version: str
    local_model_path: str | None
    local_model_present: bool | None
    local_model_size_bytes: int | None
    local_model_error: str | None
    torch_importable: bool
    torch_error: str | None
    torch_cuda_available: bool | None
    torch_cuda_device_name: str | None
    nemo_asr_importable: bool
    nemo_asr_error: str | None
    ffmpeg_path: str | None
    readiness: str


def probe_asr_environment(
    backend: str,
    model_path: str | Path | None = None,
) -> ASREnvironmentProbe:
    normalized = backend.strip().casefold()
    torch_importable, torch_error = _can_import("torch")
    nemo_importable, nemo_error = _can_import("nemo.collections.asr")
    cuda_available: bool | None = None
    cuda_device_name: str | None = None
    local_model_path: str | None = None
    local_model_present: bool | None = None
    local_model_size_bytes: int | None = None
    local_model_error: str | None = None

    if normalized == NEMOTRON_BACKEND_NAME:
        resolved_model_path = _resolve_nemotron_model_path(model_path)
        local_model_path = str(resolved_model_path)
        try:
            local_model_present = resolved_model_path.is_file()
            if local_model_present:
                local_model_size_bytes = resolved_model_path.stat().st_size
        except OSError as exc:
            local_model_present = False
            local_model_error = f"{type(exc).__name__}: {exc}"

    if torch_importable:
        try:
            import torch  # type: ignore[import-not-found]

            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_device_name = str(torch.cuda.get_device_name(0))
        except Exception as exc:  # pragma: no cover - host dependent
            torch_importable = False
            torch_error = f"{type(exc).__name__}: {exc}"

    ffmpeg_path = shutil.which("ffmpeg")
    if normalized == NEMOTRON_BACKEND_NAME:
        if not local_model_present:
            readiness = "missing model"
        elif not torch_importable or ffmpeg_path is None:
            readiness = "missing dependencies"
        elif not nemo_importable:
            readiness = "model present, missing nemo"
        elif cuda_available is not True:
            readiness = "missing cuda"
        else:
            readiness = "ready"
    elif normalized in SUPPORTED_ASR_BACKENDS:
        readiness = "ready" if normalized == FAKE_BACKEND_NAME else "missing dependencies"
    else:
        readiness = "unknown"

    return ASREnvironmentProbe(
        backend=backend,
        python_version=platform.python_version(),
        local_model_path=local_model_path,
        local_model_present=local_model_present,
        local_model_size_bytes=local_model_size_bytes,
        local_model_error=local_model_error,
        torch_importable=torch_importable,
        torch_error=torch_error,
        torch_cuda_available=cuda_available,
        torch_cuda_device_name=cuda_device_name,
        nemo_asr_importable=nemo_importable,
        nemo_asr_error=nemo_error,
        ffmpeg_path=ffmpeg_path,
        readiness=readiness,
    )


def format_asr_environment_probe(probe: ASREnvironmentProbe) -> str:
    lines = [
        f"ASR backend probe: {probe.backend}",
        f"Python version: {probe.python_version}",
    ]
    if probe.local_model_path is not None:
        lines.extend(
            [
                f"local Nemotron model path: {probe.local_model_path}",
                "local Nemotron model present: "
                f"{_unknown_yes_no(probe.local_model_present)}",
                "local Nemotron model size: "
                f"{_format_size(probe.local_model_size_bytes)}",
            ]
        )
    lines.extend(
        [
            f"torch importable: {_yes_no(probe.torch_importable)}",
            f"torch cuda available: {_unknown_yes_no(probe.torch_cuda_available)}",
            f"torch cuda device: {probe.torch_cuda_device_name or 'none'}",
            f"nemo.collections.asr importable: {_yes_no(probe.nemo_asr_importable)}",
            f"ffmpeg path: {probe.ffmpeg_path or 'not found'}",
            f"Nemotron readiness: {probe.readiness}",
        ]
    )
    if probe.local_model_error:
        lines.append(f"local model error: {probe.local_model_error}")
    if probe.torch_error:
        lines.append(f"torch error: {probe.torch_error}")
    if probe.nemo_asr_error:
        lines.append(f"nemo error: {probe.nemo_asr_error}")
    return "\n".join(lines)


def _can_import(module_name: str) -> tuple[bool, str | None]:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if spec is None:
        return False, "module not found"
    return True, None


def _normalize_nemotron_output(output: Any) -> tuple[str, str | None]:
    text = output.text if hasattr(output, "text") else str(output)
    text = text.strip()
    language_match = _NEMOTRON_LANGUAGE_TAG.search(text)
    if language_match is None:
        return text, None
    return text[: language_match.start()].rstrip(), language_match.group("language")


def _normalize_nemotron_timestamps(
    output: Any,
    *,
    chunk_start_seconds: float,
) -> dict[str, list[dict[str, object]]]:
    raw_timestamps = getattr(output, "timestamp", None)
    if not isinstance(raw_timestamps, dict):
        return {"word_timestamps": [], "segment_timestamps": []}

    normalized: dict[str, list[dict[str, object]]] = {}
    for level, text_key in (("word", "word"), ("segment", "segment")):
        entries: list[dict[str, object]] = []
        for raw_entry in raw_timestamps.get(level, []):
            if not isinstance(raw_entry, dict):
                continue
            value = raw_entry.get(text_key)
            if not isinstance(value, str) or _NEMOTRON_LANGUAGE_TAG.fullmatch(
                value.strip()
            ):
                continue
            start = raw_entry.get("start")
            end = raw_entry.get("end")
            if not isinstance(start, (int, float)) or not isinstance(
                end, (int, float)
            ):
                continue
            entries.append(
                {
                    "text": value,
                    "start_seconds": round(chunk_start_seconds + float(start), 3),
                    "end_seconds": round(chunk_start_seconds + float(end), 3),
                }
            )
        normalized[f"{level}_timestamps"] = entries
    return normalized


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _unknown_yes_no(value: bool | None) -> str:
    if value is None:
        return "unknown"
    return _yes_no(value)


def _resolve_nemotron_model_path(model_path: str | Path | None = None) -> Path:
    return Path(model_path or DEFAULT_NEMOTRON_MODEL_PATH).expanduser()


def _format_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    mib = size_bytes / (1024 * 1024)
    return f"{size_bytes} bytes ({mib:.1f} MiB)"
