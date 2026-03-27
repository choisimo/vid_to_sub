from __future__ import annotations

import unittest

from vid_to_sub_app.cli.transcription import _segments_from_mapping


class TranscriptionMetadataTests(unittest.TestCase):
    def test_segments_from_mapping_preserves_word_alignment(self) -> None:
        segments = _segments_from_mapping(
            {
                "segments": [
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello",
                        "speaker": "SPEAKER_00",
                        "words": [
                            {
                                "word": "hel",
                                "start": 0.0,
                                "end": 0.4,
                                "score": 0.9,
                            },
                            {
                                "word": "lo",
                                "start": 0.41,
                                "end": 0.6,
                            },
                        ],
                    }
                ]
            }
        )

        self.assertEqual(1, len(segments))
        self.assertEqual("SPEAKER_00", segments[0]["speaker"])
        self.assertAlmostEqual(0.6, segments[0]["word_end"])
        self.assertEqual(0.9, segments[0]["words"][0]["probability"])


if __name__ == "__main__":
    unittest.main()
