from __future__ import annotations

import json
import os
import queue
import threading
from asyncio import sleep
from dataclasses import dataclass
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from meeting_agent.events import TranscriptEvent


DEFAULT_LLM_BASE_URL = "http://127.0.0.1:18080/v1"
DEFAULT_LLM_MODEL = "gemma4-e4b-any"
DEFAULT_LLM_TIMEOUT_SECONDS = 120.0


class LLMAnalysisError(RuntimeError):
    """Raised when the local LLM cannot return a valid note analysis."""


@dataclass(frozen=True)
class NoteDraft:
    title: str
    body: str


@dataclass(frozen=True)
class LLMAnalysis:
    summary: NoteDraft | None
    concepts: tuple[NoteDraft, ...]
    questions: tuple[str, ...]


class TranscriptAnalyzer(Protocol):
    def analyze(
        self,
        events: tuple[TranscriptEvent, ...],
        *,
        create_summary: bool,
    ) -> LLMAnalysis:
        """Analyze recent transcript events in a blocking worker context."""


@dataclass(frozen=True)
class _AnalysisJob:
    events: tuple[TranscriptEvent, ...]
    create_summary: bool
    response: queue.Queue[tuple[bool, LLMAnalysis | BaseException]]


class LLMInferenceWorker:
    """Own one dedicated thread for blocking local-model inference calls."""

    def __init__(self, analyzer: TranscriptAnalyzer) -> None:
        self.analyzer = analyzer
        self._jobs: queue.Queue[_AnalysisJob | None] = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="aegis-llm-worker",
            daemon=True,
        )
        self._thread.start()

    async def analyze(
        self,
        events: tuple[TranscriptEvent, ...],
        *,
        create_summary: bool,
    ) -> LLMAnalysis:
        if self._closed:
            raise LLMAnalysisError("LLM inference worker is closed")
        response: queue.Queue[
            tuple[bool, LLMAnalysis | BaseException]
        ] = queue.Queue(maxsize=1)
        self._jobs.put(
            _AnalysisJob(
                events=events,
                create_summary=create_summary,
                response=response,
            )
        )
        while True:
            try:
                succeeded, value = response.get_nowait()
            except queue.Empty:
                await sleep(0.01)
                continue
            if succeeded:
                if not isinstance(value, LLMAnalysis):
                    raise LLMAnalysisError(
                        "LLM worker returned an invalid analysis value"
                    )
                return value
            if isinstance(value, BaseException):
                raise value
            raise LLMAnalysisError("LLM worker returned an invalid error value")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._jobs.put(None)
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            raise RuntimeError("LLM inference worker did not stop within 5 seconds")

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                if job is None:
                    return
                try:
                    result = self.analyzer.analyze(
                        job.events,
                        create_summary=job.create_summary,
                    )
                except BaseException as exc:
                    job.response.put((False, exc))
                else:
                    job.response.put((True, result))
            finally:
                self._jobs.task_done()


class LocalGemmaAnalyzer:
    """Analyze transcripts through the local llama-server OpenAI-style API."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_LLM_BASE_URL,
        model: str = DEFAULT_LLM_MODEL,
        timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_environment(cls) -> LocalGemmaAnalyzer:
        timeout_value = os.environ.get(
            "AEGIS_LLM_TIMEOUT_SECONDS",
            str(DEFAULT_LLM_TIMEOUT_SECONDS),
        )
        try:
            timeout_seconds = float(timeout_value)
        except ValueError as exc:
            raise ValueError(
                "AEGIS_LLM_TIMEOUT_SECONDS must be a number"
            ) from exc
        return cls(
            base_url=os.environ.get("AEGIS_LLM_BASE_URL", DEFAULT_LLM_BASE_URL),
            model=os.environ.get("AEGIS_LLM_MODEL", DEFAULT_LLM_MODEL),
            timeout_seconds=timeout_seconds,
        )

    def analyze(
        self,
        events: tuple[TranscriptEvent, ...],
        *,
        create_summary: bool,
    ) -> LLMAnalysis:
        if not events:
            return LLMAnalysis(summary=None, concepts=(), questions=())

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _format_analysis_request(
                        events,
                        create_summary=create_summary,
                    ),
                },
            ],
            "temperature": 0.1,
            "max_tokens": 900,
            "stream": False,
            "response_format": {
                "type": "json_schema",
                "schema": _ANALYSIS_SCHEMA,
            },
        }
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMAnalysisError(
                f"Local Gemma HTTP {exc.code}: {detail[:500]}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise LLMAnalysisError(
                f"Cannot reach local Gemma at {self.base_url}: {exc}"
            ) from exc

        return _parse_chat_completion(response_body, create_summary=create_summary)


_SYSTEM_PROMPT = """你是 Aegis，本機優先的課堂旁聽研究助理。
你的工作是根據逐字稿動態辨識真正重要的概念、形成研究筆記，並找出值得追問的未解問題。
不得依賴預設關鍵詞清單，也不要因為看到一般名詞就建立概念筆記。
只使用逐字稿中有根據的內容；資訊不足時保持空陣列，不得自行補造事實。
輸出使用繁體中文，專有名詞保留原文。只回傳符合指定 schema 的 JSON。"""


_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "anyOf": [
                {"type": "string"},
                {"type": "null"},
            ]
        },
        "concepts": {
            "type": "array",
            "items": {"type": "string"},
        },
        "questions": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["summary", "concepts", "questions"],
    "additionalProperties": False,
}


def _format_analysis_request(
    events: tuple[TranscriptEvent, ...],
    *,
    create_summary: bool,
) -> str:
    lines = []
    for index, event in enumerate(events, start=1):
        speaker = event.speaker or "未知講者"
        lines.append(f"{index}. [{speaker}] {event.text}")
    summary_instruction = (
        "請根據全部片段產生一份簡短 summary。"
        if create_summary
        else "這一輪不要產生 summary，summary 必須是 null。"
    )
    return (
        f"{summary_instruction}\n"
        "concepts 只列出本輪值得建立研究筆記的重要概念，每個元素使用「概念名稱｜為何重要的一句說明」格式；questions 只列出逐字稿尚未解答、值得研究的問題。\n\n"
        "最近逐字稿：\n"
        + "\n".join(lines)
    )


def _parse_chat_completion(
    response_body: str,
    *,
    create_summary: bool,
) -> LLMAnalysis:
    try:
        envelope = json.loads(response_body)
        content = envelope["choices"][0]["message"]["content"]
        if not isinstance(content, str):
            raise TypeError("message content is not a string")
        parsed = json.loads(_strip_json_fence(content))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise LLMAnalysisError(
            f"Local Gemma returned an invalid structured response: {exc}"
        ) from exc

    if not isinstance(parsed, dict):
        raise LLMAnalysisError("Local Gemma analysis must be a JSON object")

    try:
        raw_summary = parsed.get("summary")
        summary = _parse_note_draft(raw_summary) if create_summary else None
        concepts = _parse_note_drafts(parsed.get("concepts"))
        questions = _parse_questions(parsed.get("questions"))
    except LLMAnalysisError as exc:
        raise LLMAnalysisError(
            f"{exc}; content={content[:1000]}"
        ) from exc
    return LLMAnalysis(
        summary=summary,
        concepts=concepts,
        questions=questions,
    )


def _parse_note_draft(value: object) -> NoteDraft | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            raise LLMAnalysisError("summary must be a non-empty string")
        return NoteDraft(title="課堂摘要", body=normalized)
    if not isinstance(value, dict):
        raise LLMAnalysisError("summary must be a string, object, or null")
    title = value.get("title")
    body = value.get("body")
    if not isinstance(title, str) or not title.strip():
        raise LLMAnalysisError("note title must be a non-empty string")
    if not isinstance(body, str) or not body.strip():
        raise LLMAnalysisError("note body must be a non-empty string")
    return NoteDraft(title=title.strip(), body=body.strip())


def _parse_note_drafts(value: object) -> tuple[NoteDraft, ...]:
    if not isinstance(value, list):
        raise LLMAnalysisError("concepts must be an array")
    drafts: list[NoteDraft] = []
    for item in value:
        if isinstance(item, str):
            normalized = item.strip()
            if not normalized:
                continue
            title, separator, body = normalized.partition("｜")
            if not separator:
                title, separator, body = normalized.partition(":")
            if not separator:
                title, separator, body = normalized.partition("：")
            drafts.append(
                NoteDraft(
                    title=title.strip(),
                    body=body.strip() if separator and body.strip() else normalized,
                )
            )
            continue
        draft = _parse_note_draft(item)
        if draft is not None:
            drafts.append(draft)
    return tuple(drafts)


def _parse_questions(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise LLMAnalysisError("questions must be an array")
    questions: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise LLMAnalysisError("each question must be a string")
        normalized = item.strip()
        if normalized:
            questions.append(normalized)
    return tuple(questions)


def _strip_json_fence(content: str) -> str:
    stripped = content.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        return stripped[7:-3].strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return stripped[3:-3].strip()
    return stripped
