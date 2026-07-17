import json
import struct
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_agent.asr import ASRBackend, ASRResult
from meeting_agent.benchmark import (
    character_error_rate,
    load_asr_benchmark_manifest,
    run_asr_benchmark,
    write_asr_benchmark_report,
)
from meeting_agent.events import AudioChunk


class StaticBenchmarkBackend(ASRBackend):
    name = "static-benchmark"

    def __init__(self) -> None:
        self.validated = False

    def validate_runtime(self) -> None:
        self.validated = True

    def transcribe(
        self,
        chunk: AudioChunk,
        *,
        language: str | None = None,
    ) -> ASRResult:
        return ASRResult(
            text="今天我們介紹資訊熵",
            metadata={"detected_language": language},
        )


class ASRBenchmarkTest(unittest.TestCase):
    def test_chinese_character_error_rate_normalizes_punctuation(self) -> None:
        cer = character_error_rate("今天，上課。", "今天上科")

        self.assertEqual(cer, 0.25)

    def test_manifest_benchmark_and_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            audio_path = tmp_path / "lecture.wav"
            self._write_silent_wav(audio_path, duration_seconds=1.0)
            manifest_path = tmp_path / "benchmark.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "id": "lecture-001",
                        "audio_filepath": audio_path.name,
                        "reference": "今天我們介紹資訊熵。",
                        "language": "zh-CN",
                        "speaker": "講師",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            backend = StaticBenchmarkBackend()

            cases = load_asr_benchmark_manifest(manifest_path)
            report = run_asr_benchmark(cases, backend=backend)
            report_path = write_asr_benchmark_report(
                report,
                tmp_path / "report.json",
            )
            saved_report = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertTrue(backend.validated)
        self.assertEqual(report.case_count, 1)
        self.assertEqual(report.character_error_rate, 0.0)
        self.assertEqual(report.cases[0].speaker, "講師")
        self.assertEqual(report.cases[0].detected_language, "zh-CN")
        self.assertEqual(saved_report["backend"], "static-benchmark")
        self.assertEqual(saved_report["cases"][0]["hypothesis"], "今天我們介紹資訊熵")

    @staticmethod
    def _write_silent_wav(path: Path, duration_seconds: float) -> None:
        sample_rate = 8000
        frame_count = int(sample_rate * duration_seconds)
        frames = b"".join(struct.pack("<h", 0) for _ in range(frame_count))
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(frames)


if __name__ == "__main__":
    unittest.main()
