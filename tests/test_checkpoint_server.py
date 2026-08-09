from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from ktpu.checkpoint import verify_checkpoint
from ktpu.engine import EngineInstallation
from ktpu.engine_bootstrap import strip_unsupported_libtpu_flags
from ktpu.errors import CheckpointError
from ktpu.server import build_vllm_command


class CheckpointTests(unittest.TestCase):
    def test_remote_sha_must_match_and_tree_must_be_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            remote = root / "remote.git"
            repo = root / "repo"
            subprocess.run(["git", "init", "--bare", str(remote)], check=True)
            subprocess.run(["git", "init", "-b", "main", str(repo)], check=True)
            subprocess.run(
                ["git", "config", "user.name", "Test"], cwd=repo, check=True
            )
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repo,
                check=True,
            )
            (repo / "README.md").write_text("safe\n")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-m", "checkpoint"], cwd=repo, check=True)
            subprocess.run(
                ["git", "remote", "add", "origin", str(remote)], cwd=repo, check=True
            )
            subprocess.run(
                ["git", "push", "-u", "origin", "main"], cwd=repo, check=True
            )
            checkpoint = verify_checkpoint(repo, run_tests=False)
            self.assertEqual(
                checkpoint.commit,
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=repo, text=True
                ).strip(),
            )
            (repo / "dirty.txt").write_text("dirty")
            with self.assertRaises(CheckpointError):
                verify_checkpoint(repo, run_tests=False)


class ServerCommandTests(unittest.TestCase):
    def test_single_sequence_loopback_all_tpu_command(self) -> None:
        installation = EngineInstallation(
            engine="vllm",
            version="1",
            root=Path("/engine"),
            python=Path("/engine/bin/python"),
            executable=Path("/engine/bin/vllm"),
            installed_versions={},
            checkpoint_commit="abc",
        )
        command = build_vllm_command(
            installation,
            model="google/gemma-4-31B-it",
            host="127.0.0.1",
            port=8123,
            tensor_parallel_size=8,
            max_model_len=32768,
            enable_thinking=True,
        )
        joined = " ".join(command)
        self.assertEqual(command[0], "/engine/bin/python")
        self.assertIn("engine_bootstrap.py", command[1])
        self.assertIn("--host 127.0.0.1", joined)
        self.assertIn("--tensor-parallel-size 8", joined)
        self.assertIn("--max-num-seqs 1", joined)
        self.assertIn("--reasoning-parser gemma4", joined)
        self.assertIn("--no-enable-log-requests", joined)
        self.assertNotIn("--disable-log-requests", joined)
        self.assertNotIn("0.0.0.0", joined)

    def test_kaggle_libtpu_compatibility_flag_is_removed(self) -> None:
        value = (
            "--xla_tpu_use_dynamic_smem_negotiation=true "
            "--another_supported_flag=true"
        )
        self.assertEqual(
            strip_unsupported_libtpu_flags(value),
            "--another_supported_flag=true",
        )


if __name__ == "__main__":
    unittest.main()
