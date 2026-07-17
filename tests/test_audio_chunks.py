import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_agent.audio import (
    AudioDependencyError,
    AudioInputError,
    chunk_audio_file,
    chunk_wav_file,
)
from meeting_agent.storage import AegisStore


class AudioChunkTest(unittest.TestCase):
    def test_chunks_wav_file_and_stores_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            wav_path = tmp_path / "fixture.wav"
            self._write_silent_wav(wav_path, duration_seconds=1.2)

            db_path = tmp_path / "aegis.db"
            store = AegisStore(db_path)
            try:
                store.initialize()
                session = store.create_session(mode="audio_file", title="fixture")
                chunks = chunk_wav_file(
                    wav_path,
                    session_id=session.id,
                    output_dir=tmp_path / "chunks",
                    chunk_seconds=0.5,
                )
                for chunk in chunks:
                    store.insert_audio_chunk(chunk)

                rows = store.list_audio_chunks()
            finally:
                store.close()

            self.assertEqual(len(chunks), 3)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(Path(chunk.path).exists() for chunk in chunks))
            self.assertTrue(all(chunk.source == "file" for chunk in rows))
            self.assertTrue(all(chunk.status == "recorded" for chunk in rows))
            self.assertEqual(rows[0].original_source_path, str(wav_path))
            self.assertEqual(rows[0].sample_rate, 8000)
            self.assertEqual(rows[0].channels, 1)
            self.assertTrue(
                math.isclose(sum(row.duration_seconds for row in rows), 1.2)
            )
            self.assertEqual([row.start_seconds for row in rows], [0.0, 0.5, 1.0])
            self.assertTrue(math.isclose(rows[-1].end_seconds, 1.2))

    def test_rejects_non_wav_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(AudioInputError):
                chunk_wav_file(
                    Path(tmpdir) / "lecture.mp3",
                    session_id="session",
                    output_dir=Path(tmpdir) / "chunks",
                )

    @unittest.skipIf(shutil.which("ffmpeg") is None, "ffmpeg is not installed")
    def test_chunks_compressed_audio_with_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            mp3_path = tmp_path / "fixture.mp3"
            self._write_mp3_fixture(mp3_path, duration_seconds=1.2)

            db_path = tmp_path / "aegis.db"
            store = AegisStore(db_path)
            try:
                store.initialize()
                session = store.create_session(mode="audio_file", title="fixture")
                chunks = chunk_audio_file(
                    mp3_path,
                    session_id=session.id,
                    output_dir=tmp_path / "chunks",
                    chunk_seconds=0.5,
                )
                for chunk in chunks:
                    store.insert_audio_chunk(chunk)

                rows = store.list_audio_chunks()
            finally:
                store.close()

            self.assertEqual(len(chunks), 3)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(Path(chunk.path).exists() for chunk in chunks))
            self.assertTrue(all(chunk.path.endswith(".wav") for chunk in chunks))
            self.assertTrue(all(chunk.source == "file" for chunk in rows))
            self.assertEqual(rows[0].original_source_path, str(mp3_path))
            self.assertEqual(rows[0].sample_rate, 16000)
            self.assertEqual(rows[0].channels, 1)
            self.assertTrue(math.isclose(rows[0].start_seconds, 0.0))
            self.assertTrue(math.isclose(rows[1].start_seconds, 0.5, abs_tol=0.02))

    def test_compressed_audio_reports_missing_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mp3_path = Path(tmpdir) / "fixture.mp3"
            mp3_path.write_bytes(b"not real mp3")

            with patch("meeting_agent.audio.shutil.which", return_value=None):
                with self.assertRaisesRegex(AudioDependencyError, "ffmpeg"):
                    chunk_audio_file(
                        mp3_path,
                        session_id="session",
                        output_dir=Path(tmpdir) / "chunks",
                    )

    def test_limits_max_chunks_for_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            wav_path = tmp_path / "fixture.wav"
            self._write_silent_wav(wav_path, duration_seconds=1.2)

            chunks = chunk_audio_file(
                wav_path,
                session_id="session",
                output_dir=tmp_path / "chunks",
                chunk_seconds=0.5,
                max_chunks=2,
            )

            self.assertEqual(len(chunks), 2)

    def _write_silent_wav(self, path: Path, duration_seconds: float) -> None:
        sample_rate = 8000
        frame_count = int(sample_rate * duration_seconds)
        frames = b"".join(struct.pack("<h", 0) for _ in range(frame_count))
        with wave.open(str(path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(sample_rate)
            writer.writeframes(frames)

    def _write_mp3_fixture(self, path: Path, duration_seconds: float) -> None:
        source_wav = path.with_suffix(".wav")
        self._write_silent_wav(source_wav, duration_seconds=duration_seconds)
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_wav),
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            self.skipTest(
                f"ffmpeg could not create mp3 fixture: {result.stderr.strip()}"
            )


if __name__ == "__main__":
    unittest.main()
