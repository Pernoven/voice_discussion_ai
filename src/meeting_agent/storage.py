from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from meeting_agent.events import (
    AudioChunk,
    ListeningSession,
    OpenQuestion,
    ResearchNote,
    TranscriptEvent,
)


DEFAULT_DB_PATH = Path("data/aegis.db")


def datetime_to_db(value: datetime) -> str:
    return value.isoformat()


def datetime_from_db(value: str) -> datetime:
    return datetime.fromisoformat(value)


class AegisStore:
    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self.connection.close()

    def initialize(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                title TEXT
            );

            CREATE TABLE IF NOT EXISTS transcript_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                text TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS research_notes (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                source_transcript_event_ids TEXT NOT NULL,
                note_type TEXT NOT NULL,
                title TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS open_questions (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                source_transcript_event_ids TEXT NOT NULL,
                question TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS audio_chunks (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                path TEXT NOT NULL,
                original_source_path TEXT,
                duration_seconds REAL NOT NULL,
                sample_rate INTEGER NOT NULL,
                channels INTEGER NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS asr_transcriptions (
                id TEXT PRIMARY KEY,
                transcript_event_id TEXT NOT NULL,
                audio_chunk_id TEXT NOT NULL,
                backend TEXT NOT NULL,
                language TEXT,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (transcript_event_id) REFERENCES transcript_events(id),
                FOREIGN KEY (audio_chunk_id) REFERENCES audio_chunks(id)
            );

            CREATE INDEX IF NOT EXISTS idx_transcript_events_session_created
                ON transcript_events(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_research_notes_session_created
                ON research_notes(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_open_questions_session_created
                ON open_questions(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_audio_chunks_session_created
                ON audio_chunks(session_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_asr_transcriptions_chunk
                ON asr_transcriptions(audio_chunk_id);
            """
        )
        added_audio_start = self._ensure_column(
            "audio_chunks",
            "start_seconds",
            "REAL NOT NULL DEFAULT 0",
        )
        self._ensure_column("transcript_events", "speaker", "TEXT")
        self._ensure_column("transcript_events", "start_seconds", "REAL")
        self._ensure_column("transcript_events", "end_seconds", "REAL")
        self._ensure_column("transcript_events", "audio_chunk_id", "TEXT")
        if added_audio_start:
            self._backfill_audio_chunk_offsets()
        self._remove_duplicate_asr_rows()
        self.connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_asr_transcriptions_unique_chunk
            ON asr_transcriptions(
                audio_chunk_id,
                backend,
                COALESCE(language, '')
            )
            """
        )
        self.connection.commit()

    def _ensure_column(
        self,
        table: str,
        column: str,
        declaration: str,
    ) -> bool:
        columns = {
            row["name"]
            for row in self.connection.execute(f"PRAGMA table_info({table})")
        }
        if column in columns:
            return False
        self.connection.execute(
            f"ALTER TABLE {table} ADD COLUMN {column} {declaration}"
        )
        return True

    def _backfill_audio_chunk_offsets(self) -> None:
        session_rows = self.connection.execute(
            "SELECT DISTINCT session_id FROM audio_chunks"
        ).fetchall()
        for session_row in session_rows:
            offset = 0.0
            chunks = self.connection.execute(
                """
                SELECT id, duration_seconds
                FROM audio_chunks
                WHERE session_id = ?
                ORDER BY created_at ASC, id ASC
                """,
                (session_row["session_id"],),
            ).fetchall()
            for chunk in chunks:
                self.connection.execute(
                    "UPDATE audio_chunks SET start_seconds = ? WHERE id = ?",
                    (offset, chunk["id"]),
                )
                offset += float(chunk["duration_seconds"])

    def _remove_duplicate_asr_rows(self) -> None:
        self.connection.execute(
            """
            DELETE FROM asr_transcriptions
            WHERE rowid NOT IN (
                SELECT MIN(rowid)
                FROM asr_transcriptions
                GROUP BY audio_chunk_id, backend, COALESCE(language, '')
            )
            """
        )

    def create_session(
        self,
        mode: str = "classroom",
        title: str | None = None,
    ) -> ListeningSession:
        session = ListeningSession(mode=mode, title=title)
        self.connection.execute(
            """
            INSERT INTO sessions (id, created_at, mode, title)
            VALUES (?, ?, ?, ?)
            """,
            (
                session.id,
                datetime_to_db(session.created_at),
                session.mode,
                session.title,
            ),
        )
        self.connection.commit()
        return session

    def insert_transcript_event(
        self,
        session_id: str,
        event: TranscriptEvent,
    ) -> TranscriptEvent:
        stored_event = replace(event, session_id=session_id)
        self.connection.execute(
            """
            INSERT INTO transcript_events (
                id,
                session_id,
                text,
                speaker,
                source,
                start_seconds,
                end_seconds,
                audio_chunk_id,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                stored_event.id,
                stored_event.session_id,
                stored_event.text,
                stored_event.speaker,
                stored_event.source,
                stored_event.start_seconds,
                stored_event.end_seconds,
                stored_event.audio_chunk_id,
                datetime_to_db(stored_event.created_at),
            ),
        )
        self.connection.commit()
        return stored_event

    def insert_research_note(self, note: ResearchNote) -> None:
        self.connection.execute(
            """
            INSERT INTO research_notes (
                id,
                session_id,
                source_transcript_event_ids,
                note_type,
                title,
                body,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                note.id,
                note.session_id,
                json.dumps(list(note.source_transcript_event_ids), ensure_ascii=False),
                note.note_type,
                note.title,
                note.body,
                datetime_to_db(note.created_at),
            ),
        )
        self.connection.commit()

    def insert_asr_transcription(
        self,
        *,
        transcript_event_id: str,
        audio_chunk_id: str,
        backend: str,
        language: str | None,
        metadata: dict[str, object],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO asr_transcriptions (
                id,
                transcript_event_id,
                audio_chunk_id,
                backend,
                language,
                metadata_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                transcript_event_id,
                audio_chunk_id,
                backend,
                language,
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                datetime_to_db(datetime.now(timezone.utc)),
            ),
        )
        self.connection.commit()

    def has_asr_transcription(
        self,
        *,
        audio_chunk_id: str,
        backend: str,
        language: str | None,
    ) -> bool:
        row = self.connection.execute(
            """
            SELECT 1
            FROM asr_transcriptions
            WHERE audio_chunk_id = ?
              AND backend = ?
              AND COALESCE(language, '') = COALESCE(?, '')
            LIMIT 1
            """,
            (audio_chunk_id, backend, language),
        ).fetchone()
        return row is not None

    def list_transcript_events(
        self,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[TranscriptEvent]:
        sql = """
            SELECT
                id,
                session_id,
                text,
                speaker,
                source,
                start_seconds,
                end_seconds,
                audio_chunk_id,
                created_at
            FROM transcript_events
        """
        params: list[object] = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY created_at ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.connection.execute(sql, tuple(params)).fetchall()
        return [
            TranscriptEvent(
                id=row["id"],
                session_id=row["session_id"],
                text=row["text"],
                speaker=row["speaker"],
                source=row["source"],
                start_seconds=row["start_seconds"],
                end_seconds=row["end_seconds"],
                audio_chunk_id=row["audio_chunk_id"],
                created_at=datetime_from_db(row["created_at"]),
            )
            for row in rows
        ]

    def insert_open_question(self, question: OpenQuestion) -> None:
        self.connection.execute(
            """
            INSERT INTO open_questions (
                id,
                session_id,
                source_transcript_event_ids,
                question,
                status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                question.id,
                question.session_id,
                json.dumps(list(question.source_transcript_event_ids), ensure_ascii=False),
                question.question,
                question.status,
                datetime_to_db(question.created_at),
            ),
        )
        self.connection.commit()

    def insert_audio_chunk(self, chunk: AudioChunk) -> None:
        self.connection.execute(
            """
            INSERT INTO audio_chunks (
                id,
                session_id,
                path,
                original_source_path,
                duration_seconds,
                sample_rate,
                channels,
                source,
                start_seconds,
                created_at,
                status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chunk.id,
                chunk.session_id,
                chunk.path,
                chunk.original_source_path,
                chunk.duration_seconds,
                chunk.sample_rate,
                chunk.channels,
                chunk.source,
                chunk.start_seconds,
                datetime_to_db(chunk.created_at),
                chunk.status,
            ),
        )
        self.connection.commit()

    def list_research_notes(self, limit: int | None = None) -> list[ResearchNote]:
        sql = """
            SELECT id, session_id, source_transcript_event_ids, note_type, title, body, created_at
            FROM research_notes
            ORDER BY created_at ASC
        """
        params: tuple[int, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self.connection.execute(sql, params).fetchall()
        return [
            ResearchNote(
                id=row["id"],
                session_id=row["session_id"],
                source_transcript_event_ids=tuple(
                    json.loads(row["source_transcript_event_ids"])
                ),
                note_type=row["note_type"],
                title=row["title"],
                body=row["body"],
                created_at=datetime_from_db(row["created_at"]),
            )
            for row in rows
        ]

    def list_audio_chunks(
        self,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[AudioChunk]:
        sql = """
            SELECT
                id,
                session_id,
                path,
                original_source_path,
                duration_seconds,
                sample_rate,
                channels,
                source,
                start_seconds,
                created_at,
                status
            FROM audio_chunks
        """
        params: list[object] = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY created_at ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self.connection.execute(sql, tuple(params)).fetchall()
        return [
            AudioChunk(
                id=row["id"],
                session_id=row["session_id"],
                path=row["path"],
                original_source_path=row["original_source_path"],
                duration_seconds=row["duration_seconds"],
                sample_rate=row["sample_rate"],
                channels=row["channels"],
                source=row["source"],
                start_seconds=row["start_seconds"],
                created_at=datetime_from_db(row["created_at"]),
                status=row["status"],
            )
            for row in rows
        ]

    def list_open_questions(self, limit: int | None = None) -> list[OpenQuestion]:
        sql = """
            SELECT id, session_id, source_transcript_event_ids, question, status, created_at
            FROM open_questions
            ORDER BY created_at ASC
        """
        params: tuple[int, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        rows = self.connection.execute(sql, params).fetchall()
        return [
            OpenQuestion(
                id=row["id"],
                session_id=row["session_id"],
                source_transcript_event_ids=tuple(
                    json.loads(row["source_transcript_event_ids"])
                ),
                question=row["question"],
                status=row["status"],
                created_at=datetime_from_db(row["created_at"]),
            )
            for row in rows
        ]
