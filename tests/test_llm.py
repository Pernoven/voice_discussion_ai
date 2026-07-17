import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_agent.events import TranscriptEvent
from meeting_agent.llm import LocalGemmaAnalyzer


class FakeHTTPResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self) -> bytes:
        return json.dumps(self.body, ensure_ascii=False).encode("utf-8")


class LocalGemmaAnalyzerTest(unittest.TestCase):
    def test_sends_schema_request_and_parses_dynamic_analysis(self) -> None:
        completion = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "討論複雜系統的湧現現象。",
                                "concepts": [
                                    "湧現行為｜由局部互動形成整體模式。"
                                ],
                                "questions": ["什麼條件會形成穩定的湧現模式？"],
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }

        with patch(
            "meeting_agent.llm.urlopen",
            return_value=FakeHTTPResponse(completion),
        ) as mocked_urlopen:
            analyzer = LocalGemmaAnalyzer(
                base_url="http://127.0.0.1:18080/v1",
                model="gemma4-e4b-any",
            )
            analysis = analyzer.analyze(
                (TranscriptEvent(text="今天討論湧現行為", speaker="講師"),),
                create_summary=True,
            )

        request = mocked_urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "gemma4-e4b-any")
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        self.assertIn("不得依賴預設關鍵詞清單", payload["messages"][0]["content"])
        self.assertEqual(analysis.summary.title, "課堂摘要")
        self.assertEqual(analysis.concepts[0].title, "湧現行為")
        self.assertEqual(
            analysis.questions,
            ("什麼條件會形成穩定的湧現模式？",),
        )

    def test_ignores_unrequested_summary(self) -> None:
        completion = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": {
                                    "title": "不應保存",
                                    "body": "模型沒有遵守指令。",
                                },
                                "concepts": [],
                                "questions": [],
                            },
                            ensure_ascii=False,
                        )
                    }
                }
            ]
        }
        with patch(
            "meeting_agent.llm.urlopen",
            return_value=FakeHTTPResponse(completion),
        ):
            analysis = LocalGemmaAnalyzer().analyze(
                (TranscriptEvent(text="單一片段"),),
                create_summary=False,
            )

        self.assertIsNone(analysis.summary)


if __name__ == "__main__":
    unittest.main()
