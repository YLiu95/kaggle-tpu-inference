from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path


class RecoveryTests(unittest.TestCase):
    def test_recovery_scripts_are_executable_and_valid_shell(self) -> None:
        root = Path(__file__).resolve().parents[1]
        scripts = [root / "scripts/checkpoint.sh", root / "scripts/resume.sh"]
        for script in scripts:
            self.assertTrue(os.access(script, os.X_OK), script)
            subprocess.run(["bash", "-n", str(script)], check=True)
        result = subprocess.run(
            ["bash", str(root / "scripts/resume.sh"), "--help"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
        self.assertIn("--clear-tpu", result.stdout)
        self.assertIn("--setup", result.stdout)

    def test_resume_documents_cache_tests_and_checkpoint(self) -> None:
        root = Path(__file__).resolve().parents[1]
        text = (root / "scripts/resume.sh").read_text(encoding="utf-8")
        self.assertIn(".cache/huggingface", text)
        self.assertIn("unittest discover", text)
        self.assertIn("checkpoint.sh", text)
        self.assertIn("ktpu status", text)


if __name__ == "__main__":
    unittest.main()

