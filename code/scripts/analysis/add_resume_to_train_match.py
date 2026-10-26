#!/usr/bin/env python3
"""Add safe checkpoint-resume support to the existing train_match.py.

Place this file in the MATCH-Reproduction project root, beside train_match.py,
then run:

    python add_resume_to_train_match.py

The script:
  * creates train_match_before_resume.py as a backup;
  * adds a --resume command-line option;
  * restores online networks, target networks, optimizer, episode, and best POL;
  * interprets --episodes as the final total episode number;
  * appends to an existing training_log.csv instead of erasing it.
"""

from __future__ import annotations

import py_compile
import shutil
from pathlib import Path


TARGET = Path("train_match.py")
BACKUP = Path("train_match_before_resume.py")


def replace_once(source: str, old: str, new: str, description: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Cannot apply {description}: expected exactly one matching block, "
            f"but found {count}. The original file was not changed."
        )
    return source.replace(old, new, 1)


def main() -> None:
    if not TARGET.is_file():
        raise FileNotFoundError(
            "train_match.py was not found. Put this script in the same "
            "MATCH-Reproduction directory as train_match.py."
        )

    original = TARGET.read_text(encoding="utf-8")

    if "--resume" in original and "Resume from episode:" in original:
        print("train_match.py already contains the resume changes; no action needed.")
        py_compile.compile(str(TARGET), doraise=True)
        print("Syntax check passed.")
        return

    modified = original

    parser_anchor = """    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--episodes",
"""
    parser_replacement = """    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from a checkpoint",
    )

    parser.add_argument(
        "--episodes",
"""
    modified = replace_once(
        modified,
        parser_anchor,
        parser_replacement,
        "the --resume argument",
    )

    resume_anchor = """    best_evaluation_pol = -1.0

    latest_training = {
"""
    resume_replacement = """    best_evaluation_pol = -1.0
    start_episode = 1

    if args.resume is not None:
        resume_path = Path(args.resume)

        if not resume_path.is_file():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {resume_path}"
            )

        print(f"Loading checkpoint: {resume_path}")

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )

        required_keys = {
            "episode",
            "match_agent",
            "information_hub",
            "mixer",
            "target_match_agent",
            "target_information_hub",
            "target_mixer",
            "optimizer",
        }
        missing_keys = sorted(required_keys.difference(checkpoint))
        if missing_keys:
            raise KeyError(
                "Resume checkpoint is missing required keys: "
                + ", ".join(missing_keys)
            )

        match_agent.load_state_dict(
            checkpoint["match_agent"]
        )
        information_hub.load_state_dict(
            checkpoint["information_hub"]
        )
        mixer.load_state_dict(
            checkpoint["mixer"]
        )

        learner.target_match_agent.load_state_dict(
            checkpoint["target_match_agent"]
        )
        learner.target_information_hub.load_state_dict(
            checkpoint["target_information_hub"]
        )
        learner.target_mixer.load_state_dict(
            checkpoint["target_mixer"]
        )
        learner.optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        completed_episode = int(
            checkpoint["episode"]
        )
        start_episode = completed_episode + 1
        best_evaluation_pol = float(
            checkpoint.get(
                "best_evaluation_pol",
                -1.0,
            )
        )

        print(
            f"Checkpoint episode: {completed_episode}"
        )
        print(
            f"Resume from episode: {start_episode}"
        )
        print(
            "Previous best evaluation POL: "
            f"{best_evaluation_pol:.4f}"
        )

        if start_episode > args.episodes:
            raise ValueError(
                f"Checkpoint is already at episode {completed_episode}, "
                f"but --episodes is {args.episodes}."
            )

    latest_training = {
"""
    modified = replace_once(
        modified,
        resume_anchor,
        resume_replacement,
        "checkpoint loading",
    )

    csv_anchor = """    with csv_path.open(
        "w",
        newline="",
    ) as csv_file:
"""
    csv_replacement = """    csv_already_exists = (
        csv_path.exists()
        and csv_path.stat().st_size > 0
    )

    with csv_path.open(
        "a",
        newline="",
    ) as csv_file:
"""
    modified = replace_once(
        modified,
        csv_anchor,
        csv_replacement,
        "CSV append mode",
    )

    modified = replace_once(
        modified,
        """        writer.writeheader()

        for episode in range(
            1,
            args.episodes + 1,
        ):
""",
        """        if not csv_already_exists:
            writer.writeheader()

        for episode in range(
            start_episode,
            args.episodes + 1,
        ):
""",
        "CSV header and resumed episode range",
    )

    # Compile a temporary candidate before touching the original file.
    candidate = TARGET.with_suffix(".resume_candidate.py")
    candidate.write_text(modified, encoding="utf-8")
    try:
        py_compile.compile(str(candidate), doraise=True)
    finally:
        candidate.unlink(missing_ok=True)

    if BACKUP.exists():
        if BACKUP.read_bytes() != TARGET.read_bytes():
            raise FileExistsError(
                f"{BACKUP} already exists and differs from train_match.py. "
                "Rename or preserve that backup, then run this script again."
            )
    else:
        shutil.copy2(TARGET, BACKUP)

    TARGET.write_text(modified, encoding="utf-8")
    py_compile.compile(str(TARGET), doraise=True)

    print("Resume support added successfully.")
    print(f"Original backup: {BACKUP}")
    print("Syntax check passed.")
    print("Next check: python train_match.py --help")


if __name__ == "__main__":
    main()

