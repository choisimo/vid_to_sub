from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from db import Database
from vid_to_sub_app.cli.output import segments_to_tsv
from vid_to_sub_app.cli.subtitle_copy import (
    is_subtitle_output_path,
    subtitle_paths_from_output_paths,
)
from vid_to_sub_app.tui.helpers import filter_subtitle_paths


class OutputRegressionTests(unittest.TestCase):
    def test_segments_to_tsv_keeps_three_columns_consistent_with_header(self) -> None:
        payload = segments_to_tsv(
            [{"start": 1.25, "end": 2.5, "text": "hello world"}]
        ).splitlines()

        self.assertEqual("start\tend\ttext", payload[0])
        self.assertEqual("1.250\t2.500\thello world", payload[1])
        self.assertEqual(3, len(payload[1].split("\t")))

    def test_subtitle_filters_exclude_json_and_stage1_artifacts(self) -> None:
        paths = [
            "/tmp/movie.srt",
            "/tmp/movie.vtt",
            "/tmp/movie.tsv",
            "/tmp/movie.txt",
            "/tmp/movie.json",
            "/tmp/movie.stage1.json",
        ]

        self.assertEqual(
            ["/tmp/movie.srt", "/tmp/movie.vtt", "/tmp/movie.tsv", "/tmp/movie.txt"],
            filter_subtitle_paths(paths),
        )
        self.assertTrue(is_subtitle_output_path("/tmp/movie.srt"))
        self.assertFalse(is_subtitle_output_path("/tmp/movie.json"))
        self.assertFalse(is_subtitle_output_path("/tmp/movie.stage1.json"))

    def test_subtitle_paths_from_output_paths_uses_same_policy(self) -> None:
        raw = json.dumps(
            [
                "/tmp/movie.srt",
                "/tmp/movie.tsv",
                "/tmp/movie.json",
                "/tmp/movie.stage1.json",
            ]
        )

        result = subtitle_paths_from_output_paths(raw)

        self.assertEqual(["movie.srt", "movie.tsv"], [path.name for path in result])

    def test_database_close_releases_connections_and_allows_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            db = Database(Path(tmpdir) / "state.db")
            first_conn = db._conn()
            db.set("sample", "value")

            db.close()

            second_conn = db._conn()
            self.assertIsNot(first_conn, second_conn)
            self.assertEqual("value", db.get("sample"))
            db.close()


if __name__ == "__main__":
    _ = unittest.main()
