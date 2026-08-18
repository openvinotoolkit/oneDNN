#!/usr/bin/python3

# *******************************************************************************
# Copyright 2024 Arm Limited and affiliates.
# Copyright 2024-2026 Intel Corporation
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

import argparse
from pathlib import Path
import re
import subprocess
import sys


def _run_git(args):
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            check=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        details = error.stderr.strip() or error.stdout.strip()
        raise RuntimeError(details) from error


def _load_grandfathered_tips(path, commits_in_range):
    if path is None:
        return []

    tips = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RuntimeError(
            f"Cannot read grandfathered history file '{path}': {error}"
        ) from error

    for line_number, line in enumerate(lines, 1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue

        fields = entry.split(maxsplit=1)
        if len(fields) != 2 or not re.fullmatch(r"[0-9a-f]{40}", fields[0]):
            raise RuntimeError(
                f"Invalid grandfathered history entry at {path}:{line_number}. "
                "Expected: <40-character commit SHA> <reason>"
            )

        commit, reason = fields
        if commit not in commits_in_range:
            print(
                f"Grandfathered history tip inactive: {commit} is not "
                "present in the checked commit range."
            )
            continue

        print(f"Grandfathered history tip: {commit} ({reason})")
        tips.append(commit)

    return tips

# Ensure the scope ends in a colon and that same level scopes are
# comma delimited.
# Current implementation only checks the first level scope as ':' can be used
# in the commit description (ex: TBB::tbb or bf16:bf16).
# TODO: Limit scopes to an acceptable list of tags.
def __scopeCheck(msg: str):
    status = "Message scope: "

    if not re.match(r"^[a-z0-9_]+(, [a-z0-9_]+)*: ", msg):
        if re.match(r"^\s+", msg):
            print(
                f"{status} FAILED: Commit message shouldn't have leading spaces"
            )
            return False

        if re.match(r"^Merge ", msg):
            print(f"{status} FAILED: Merge commits are not allowed")
            return False

        print(
            f"{status} FAILED: Commit message must follow the format "
            "<scope>:[ <scope>:] <short description>"
        )
        return False

    print(f"{status} OK")
    return True

# Ensure a character limit for the first line.
def __numCharacterCheck(msg: str):
    status = "Message length:"
    if len(msg) <= 72:
        print(f"{status} OK")
        return True
    else:
        # Fixup or revert commits usually include the full name of the commit
        # they are fixing, which adds 6 more symbols to the message.
        # Let them in.
        if re.match(r"^fixup: ", msg):
            print(f"{status} Fixup message, OK")
            return True
        elif re.match(r"^revert: ", msg):
            print(f"{status} Revert message, OK")
            return True
        else:
            print(
                f"{status} FAILED: Commit message summary must not "
                "exceed 72 characters."
            )
            return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("head", help="Head commit of PR branch")
    parser.add_argument("base", help="Base commit of PR branch")
    parser.add_argument(
        "--grandfathered",
        metavar="FILE",
        help=(
            "File containing immutable history tips to exclude. Each line "
            "must contain a full commit SHA and an audit reason."
        ),
    )
    args = parser.parse_args()
    base: str = args.base
    head: str = args.head

    commit_range = base + ".." + head
    try:
        all_messages = _run_git(
            ["rev-list", "--format=oneline", commit_range]
        ).splitlines()
        commits_in_range = {line.split(" ", 1)[0] for line in all_messages}
        grandfathered_tips = _load_grandfathered_tips(
            args.grandfathered, commits_in_range
        )
        revision_args = ["rev-list", "--format=oneline", commit_range]
        if grandfathered_tips:
            revision_args.extend(["--not", *grandfathered_tips])
        messages = _run_git(revision_args).splitlines()
    except RuntimeError as error:
        print(f"Commit message check setup FAILED: {error}", file=sys.stderr)
        return 2

    skipped_count = len(all_messages) - len(messages)
    if skipped_count:
        print(
            f"Skipped {skipped_count} commits from explicitly "
            "grandfathered history."
        )

    is_ok = True
    for i in messages:
        print(i)
        commit_msg = i.split(" ", 1)[1]
        result = __numCharacterCheck(commit_msg)
        is_ok = is_ok and result
        result = __scopeCheck(commit_msg)
        is_ok = is_ok and result

    if is_ok:
        print("All commit messages are formatted correctly.")
    else:
        print(
            "Some commit message checks failed. Please align commit messages "
            "with Contributing Guidelines and update the PR."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
