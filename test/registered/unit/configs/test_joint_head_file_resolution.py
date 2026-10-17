import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sglang.srt.configs.model_config import (
    load_joint_head_config,
    resolve_joint_head_file,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestJointHeadFileResolution(CustomTestCase):
    def setUp(self):
        load_joint_head_config.cache_clear()

    def tearDown(self):
        load_joint_head_config.cache_clear()

    def _write_config(self, directory: Path, config: dict) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "joint_head_config.json"
        path.write_text(json.dumps(config))
        return path

    def test_finds_config_in_clef_head(self):
        expected = {"source": "clef_head"}
        with TemporaryDirectory() as checkpoint:
            path = self._write_config(Path(checkpoint) / "clef_head", expected)

            self.assertEqual(
                load_joint_head_config(checkpoint, None),
                expected,
            )
            self.assertEqual(
                resolve_joint_head_file(checkpoint, None, "joint_head_config.json"),
                str(path),
            )

    def test_prefers_root_config_when_both_paths_exist(self):
        expected = {"source": "root"}
        with TemporaryDirectory() as checkpoint:
            root_path = self._write_config(Path(checkpoint), expected)
            self._write_config(Path(checkpoint) / "clef_head", {"source": "clef_head"})

            self.assertEqual(load_joint_head_config(checkpoint, None), expected)
            self.assertEqual(
                resolve_joint_head_file(checkpoint, None, "joint_head_config.json"),
                str(root_path),
            )

    def test_returns_none_when_config_is_missing(self):
        with TemporaryDirectory() as checkpoint:
            self.assertIsNone(load_joint_head_config(checkpoint, None))
            self.assertIsNone(
                resolve_joint_head_file(checkpoint, None, "joint_head_config.json")
            )


if __name__ == "__main__":
    unittest.main()
