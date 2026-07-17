import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from meeting_agent.brain import MeetingBrain
from meeting_agent.config import AgentConfig
from meeting_agent.events import TranscriptEvent


class MeetingBrainTest(unittest.TestCase):
    def test_legacy_brain_does_not_use_fixed_keyword_triggers(self) -> None:
        brain = MeetingBrain(AgentConfig(summary_every_turns=8))

        utterance = brain.process(TranscriptEvent(text="今天討論 Entropy"))

        self.assertIsNone(utterance)

    def test_manual_summary_uses_recent_transcript(self) -> None:
        brain = MeetingBrain(AgentConfig())
        brain.process(TranscriptEvent(text="第一段討論"))

        utterance = brain.process(TranscriptEvent(text="/summary"))

        self.assertIsNotNone(utterance)
        assert utterance is not None
        self.assertEqual(utterance.reason, "manual_summary")
        self.assertIn("第一段討論", utterance.text)


if __name__ == "__main__":
    unittest.main()
