#!/usr/bin/env python3

# *******************************************************************************
# Copyright 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# *******************************************************************************

import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("commit-msg-check.py")


class CommitMessageCheckTest(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.repo = Path(self._temp_dir.name)
        self._git("init", "--quiet")
        self._git("config", "user.email", "ci@example.com")
        self._git("config", "user.name", "CI Test")
        self.base = self._commit("common: initialize test history")

    def tearDown(self):
        self._temp_dir.cleanup()

    def _git(self, *args):
        return subprocess.run(
            ["git", *args],
            cwd=self.repo,
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()

    def _commit(self, message):
        self._git("commit", "--allow-empty", "--quiet", "-m", message)
        return self._git("rev-parse", "HEAD")

    def _run_check(self, head, grandfathered=None):
        command = ["python3", str(SCRIPT), head, self.base]
        if grandfathered is not None:
            command.extend(["--grandfathered", str(grandfathered)])
        return subprocess.run(
            command,
            cwd=self.repo,
            capture_output=True,
            text=True,
        )

    def _write_grandfathered(self, contents):
        path = self.repo / "grandfathered.txt"
        path.write_text(contents, encoding="utf-8")
        return path

    def test_checks_entire_range_without_grandfathered_history(self):
        head = self._commit("[FIX] legacy subject")

        result = self._run_check(head)

        self.assertEqual(result.returncode, 1)
        self.assertIn("Message scope:  FAILED", result.stdout)

    def test_grandfathered_tip_skips_ancestors(self):
        legacy_tip = self._commit("[FIX] legacy subject")
        head = self._commit("ci: validate new descendant")
        grandfathered = self._write_grandfathered(
            f"{legacy_tip} imported immutable history\n"
        )

        result = self._run_check(head, grandfathered)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Skipped 1 commits", result.stdout)
        self.assertIn("ci: validate new descendant", result.stdout)
        self.assertNotIn("[FIX] legacy subject", result.stdout)

    def test_bad_descendant_is_not_grandfathered(self):
        legacy_tip = self._commit("[FIX] legacy subject")
        head = self._commit("[FIX] new bad subject")
        grandfathered = self._write_grandfathered(
            f"{legacy_tip} imported immutable history\n"
        )

        result = self._run_check(head, grandfathered)

        self.assertEqual(result.returncode, 1)
        self.assertIn("[FIX] new bad subject", result.stdout)
        self.assertIn("Message scope:  FAILED", result.stdout)

    def test_malformed_entry_fails_closed(self):
        head = self._commit("ci: validate descendant")
        grandfathered = self._write_grandfathered("not-a-sha migration\n")

        result = self._run_check(head, grandfathered)

        self.assertEqual(result.returncode, 2)
        self.assertIn("Invalid grandfathered history entry", result.stderr)

    def test_unknown_tip_does_not_exclude_bad_descendant(self):
        head = self._commit("[FIX] new bad subject")
        grandfathered = self._write_grandfathered(
            f"{'0' * 40} missing migration tip\n"
        )

        result = self._run_check(head, grandfathered)

        self.assertEqual(result.returncode, 1)
        self.assertIn("tip inactive", result.stdout)
        self.assertIn("[FIX] new bad subject", result.stdout)


if __name__ == "__main__":
    unittest.main()
