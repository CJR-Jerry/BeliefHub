#!/usr/bin/env python3
"""
Safely prepare and validate the MATCH CN6 entry points.

Place this file in the MATCH-Reproduction project root and run:

    python setup_cn6.py

By default the script:
1. creates/repairs the three CN6 entry-point files without touching CN5;
2. backs up the pre-existing CN6 files;
3. checks the CN6 environment and tensor dimensions;
4. runs a 10-episode smoke training and saves its console log.

Use --skip-smoke to apply the fix and run checks without training.
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import os
import py_compile
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


N_AGENTS = 6
N_LANDMARKS = 6
N_ENTITIES = N_AGENTS + N_LANDMARKS
EXPECTED_OBSERVATION_DIM = 36
EXPECTED_STATE_DIM = 216
EXPECTED_ENTITY_DIM = 8
EXPECTED_ACTIONS = 5

ENTRY_POINTS: Dict[str, str] = {
    "train_match_cn6.py": "train_match.py",
    "resume_match_cn6.py": "resume_match.py",
    "evaluate_match_cn6.py": "evaluate_match.py",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair CN6 dimensions, validate the setup, and optionally "
            "run a 10-episode smoke training."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "MATCH-Reproduction root. Defaults to the directory containing "
            "this script."
        ),
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Apply the fix and run checks, but do not start smoke training.",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help=(
            "Only repair files and run static checks. This is intended for "
            "inspection on a machine without the MATCH Python environment."
        ),
    )
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help=(
            "Optional CUDA_VISIBLE_DEVICES value for smoke training, "
            "for example 0 or 3. By default the current environment is used."
        ),
    )
    parser.add_argument(
        "--smoke-output",
        type=Path,
        default=None,
        help=(
            "Optional smoke-test output directory relative to the project "
            "root (or an absolute path). A timestamped CN6 directory is used "
            "by default."
        ),
    )
    return parser.parse_args()


def require_project(project_root: Path) -> None:
    required = [
        project_root / "src",
        project_root / "src" / "envs" / "cooperative_navigation.py",
        project_root / "src" / "envs" / "entity_observation.py",
        project_root / "src" / "models" / "match_information_hub.py",
        project_root / "src" / "runners" / "match_episode_runner.py",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "This does not look like the MATCH-Reproduction project root. "
            "Missing:\n  - " + "\n  - ".join(missing)
        )

    missing_pairs = [
        f"{target} (and source {source})"
        for target, source in ENTRY_POINTS.items()
        if not (project_root / target).is_file()
        and not (project_root / source).is_file()
    ]
    if missing_pairs:
        raise FileNotFoundError(
            "Cannot create or repair these CN6 entry points:\n  - "
            + "\n  - ".join(missing_pairs)
        )


def ensure_cn6_files(project_root: Path) -> List[Path]:
    paths: List[Path] = []
    for target_name, source_name in ENTRY_POINTS.items():
        target = project_root / target_name
        source = project_root / source_name
        if not target.exists():
            shutil.copy2(source, target)
            print(f"[created] {target_name} from {source_name}")
        paths.append(target)
    return paths


def make_backup(project_root: Path, paths: Iterable[Path]) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / "cn6_backups" / stamp
    suffix = 1
    while backup_dir.exists():
        backup_dir = project_root / "cn6_backups" / f"{stamp}_{suffix}"
        suffix += 1
    backup_dir.mkdir(parents=True)
    for path in paths:
        shutil.copy2(path, backup_dir / path.name)
    print(f"[backup] {backup_dir.relative_to(project_root)}")
    return backup_dir


def replace_assignment(text: str, name: str, replacement: str) -> Tuple[str, int]:
    pattern = re.compile(
        rf"^(?P<indent>[ \t]*){re.escape(name)}"
        rf"(?:[ \t]*:[^=\n]+)?[ \t]*=[ \t]*"
        rf"(?:5|6|10|12|n_agents[ \t]*\+[ \t]*n_landmarks)"
        rf"(?P<tail>[ \t]*(?:#.*)?$)",
        flags=re.MULTILINE,
    )
    return pattern.subn(
        lambda match: (
            f"{match.group('indent')}{name} = {replacement}"
            f"{match.group('tail')}"
        ),
        text,
    )


def repair_entry_point(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    text = original.replace("match_cn5", "match_cn6")

    text, agent_count = replace_assignment(text, "n_agents", "6")
    text, landmark_count = replace_assignment(text, "n_landmarks", "6")
    text, entity_count = replace_assignment(
        text,
        "n_entities",
        "n_agents + n_landmarks",
    )

    text = re.sub(
        r"(?m)^(?P<indent>[ \t]*)print\("
        r"[\"']Agents/Landmarks:[ \t]*5/5[\"']"
        r"\)[ \t]*$",
        (
            r"\g<indent>print("
            r'f"Agents/Landmarks: {n_agents}/{n_landmarks}"'
            r")"
        ),
        text,
    )

    if path.name == "train_match_cn6.py":
        text = re.sub(
            r"(?m)^(?P<indent>[ \t]*)n_agents[ \t]*=[ \t]*5,"
            r"[ \t]*(?P<tail>#.*)?$",
            r"\g<indent>n_agents=6,\g<tail>",
            text,
        )

    if agent_count == 0:
        raise RuntimeError(f"No n_agents assignment found in {path.name}.")
    if landmark_count == 0:
        raise RuntimeError(f"No n_landmarks assignment found in {path.name}.")
    if entity_count == 0:
        raise RuntimeError(f"No n_entities assignment found in {path.name}.")

    if text != original:
        path.write_text(text, encoding="utf-8")
        print(f"[fixed] {path.name}")
        return True

    print(f"[ok] {path.name} was already repaired")
    return False


def assignment_values(path: Path) -> Dict[str, List[object]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: Dict[str, List[object]] = {
        "n_agents": [],
        "n_landmarks": [],
        "n_entities": [],
    }

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue

        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value_node = node.value

        for target in targets:
            if not isinstance(target, ast.Name) or target.id not in values:
                continue
            try:
                value = ast.literal_eval(value_node)
            except (ValueError, TypeError):
                if (
                    isinstance(value_node, ast.BinOp)
                    and isinstance(value_node.op, ast.Add)
                    and isinstance(value_node.left, ast.Name)
                    and value_node.left.id == "n_agents"
                    and isinstance(value_node.right, ast.Name)
                    and value_node.right.id == "n_landmarks"
                ):
                    value = "n_agents + n_landmarks"
                else:
                    value = ast.dump(value_node)
            values[target.id].append(value)
    return values


def static_checks(project_root: Path, paths: Iterable[Path]) -> None:
    errors: List[str] = []
    for path in paths:
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"{path.name}: Python syntax error: {exc.msg}")
            continue

        text = path.read_text(encoding="utf-8")
        values = assignment_values(path)

        if "match_cn5" in text:
            errors.append(f"{path.name}: still contains match_cn5")
        if 6 not in values["n_agents"]:
            errors.append(f"{path.name}: n_agents is not configured as 6")
        if 6 not in values["n_landmarks"]:
            errors.append(f"{path.name}: n_landmarks is not configured as 6")
        if "n_agents + n_landmarks" not in values["n_entities"]:
            errors.append(
                f"{path.name}: n_entities is not computed as "
                "n_agents + n_landmarks"
            )
        if 10 in values["n_entities"]:
            errors.append(f"{path.name}: still contains n_entities = 10")

    train_text = (project_root / "train_match_cn6.py").read_text(
        encoding="utf-8"
    )
    if "Agents/Landmarks: 5/5" in train_text:
        errors.append("train_match_cn6.py: status print still says 5/5")
    if not re.search(r"n_agents[ \t]*=[ \t]*6,", train_text):
        errors.append(
            "train_match_cn6.py: CooperativeNavigation is not passed "
            "n_agents=6"
        )

    if errors:
        raise RuntimeError("Static CN6 checks failed:\n  - " + "\n  - ".join(errors))

    print("[check] Python syntax and CN6 entry-point configuration passed")


def runtime_dimension_check(project_root: Path) -> None:
    sys.path.insert(0, str(project_root))
    env = None
    try:
        from src.envs.cooperative_navigation import CooperativeNavigation
        from src.envs.entity_observation import MPEEntityObservationAdapter

        env = CooperativeNavigation(
            n_agents=N_AGENTS,
            max_cycles=25,
            occupancy_radius=0.3,
        )
        observations, _ = env.reset(seed=0)
        first_agent = env.possible_agents[0]
        observation_dim = env.observation_space(first_agent).shape[0]
        state_dim = env.env.state().size
        n_actions = env.action_space(first_agent).n

        adapter = MPEEntityObservationAdapter(
            n_agents=N_AGENTS,
            n_landmarks=N_LANDMARKS,
            sight_range=None,
        )
        entity_observations, observation_mask = adapter.convert(
            observations=observations,
            agent_names=env.possible_agents,
            packet_loss=0.0,
        )

        actual = {
            "agents": len(env.possible_agents),
            "observation_dim": observation_dim,
            "state_dim": state_dim,
            "n_entities": adapter.n_entities,
            "entity_observations": tuple(entity_observations.shape),
            "observation_mask": tuple(observation_mask.shape),
            "n_actions": n_actions,
        }
        expected = {
            "agents": N_AGENTS,
            "observation_dim": EXPECTED_OBSERVATION_DIM,
            "state_dim": EXPECTED_STATE_DIM,
            "n_entities": N_ENTITIES,
            "entity_observations": (
                N_AGENTS,
                N_ENTITIES,
                EXPECTED_ENTITY_DIM,
            ),
            "observation_mask": (N_AGENTS, N_ENTITIES),
            "n_actions": EXPECTED_ACTIONS,
        }
        mismatches = [
            f"{key}: got {actual[key]!r}, expected {expected[key]!r}"
            for key in expected
            if actual[key] != expected[key]
        ]
        if mismatches:
            raise RuntimeError(
                "Runtime CN6 dimension check failed:\n  - "
                + "\n  - ".join(mismatches)
            )

        print(
            "[check] Runtime dimensions passed: "
            "agents=6, landmarks=6, obs=36, state=216, entities=12, "
            "entity_features=8, actions=5"
        )
    finally:
        if env is not None:
            env.close()
        if sys.path and sys.path[0] == str(project_root):
            sys.path.pop(0)


def resolve_smoke_paths(
    project_root: Path,
    requested_output: Path | None,
) -> Tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if requested_output is None:
        output_dir = (
            project_root
            / "outputs"
            / f"match_cn6_smoke_{stamp}_seed0"
        )
    elif requested_output.is_absolute():
        output_dir = requested_output
    else:
        output_dir = project_root / requested_output

    if output_dir.exists():
        raise FileExistsError(
            f"Smoke output already exists; refusing to overwrite: {output_dir}"
        )

    log_path = project_root / f"cn6_smoke_{stamp}_seed0.log"
    return output_dir, log_path


def run_and_tee(
    command: List[str],
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
) -> None:
    print("[smoke] " + " ".join(command))
    print(f"[smoke] console log: {log_path.name}")

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()
        return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"CN6 smoke training failed with exit code {return_code}. "
            f"See {log_path}."
        )


def validate_smoke_output(output_dir: Path, log_path: Path) -> None:
    training_log = output_dir / "training_log.csv"
    latest_checkpoint = output_dir / "checkpoint_latest.pt"

    missing = [
        str(path)
        for path in (training_log, latest_checkpoint, log_path)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "Smoke training finished but expected artifacts are missing:\n  - "
            + "\n  - ".join(missing)
        )

    with training_log.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 10:
        raise RuntimeError(
            f"Expected 10 training rows, found {len(rows)} in {training_log}."
        )

    finite_losses = []
    evaluation_episodes = []
    for row in rows:
        try:
            loss = float(row["loss"])
            if math.isfinite(loss):
                finite_losses.append(loss)
        except (KeyError, TypeError, ValueError):
            pass

        try:
            evaluation_pol = float(row["evaluation_pol"])
            if math.isfinite(evaluation_pol):
                evaluation_episodes.append(int(row["episode"]))
        except (KeyError, TypeError, ValueError):
            pass

    if not finite_losses:
        raise RuntimeError("Smoke training produced no finite training loss.")
    if not {5, 10}.issubset(set(evaluation_episodes)):
        raise RuntimeError(
            "Expected evaluations at episodes 5 and 10, but found them at "
            f"{evaluation_episodes}."
        )

    console = log_path.read_text(encoding="utf-8", errors="replace")
    required_console_lines = [
        "Agents/Landmarks: 6/6",
        "Observation dimension: 36",
        "State dimension: 216",
        "Entity count: 12",
    ]
    absent = [line for line in required_console_lines if line not in console]
    if absent:
        raise RuntimeError(
            "Smoke console output is missing expected CN6 configuration:\n  - "
            + "\n  - ".join(absent)
        )

    print(
        "[passed] 10-episode CN6 smoke training completed with finite loss "
        "and evaluations at episodes 5 and 10"
    )
    print(f"[result] output: {output_dir}")
    print(f"[result] log: {log_path}")


def run_smoke(
    project_root: Path,
    gpu: str | None,
    requested_output: Path | None,
) -> None:
    output_dir, log_path = resolve_smoke_paths(
        project_root,
        requested_output,
    )
    command = [
        sys.executable,
        "-u",
        "train_match_cn6.py",
        "--episodes",
        "10",
        "--seed",
        "0",
        "--batch-size",
        "4",
        "--buffer-capacity",
        "50",
        "--warmup-episodes",
        "4",
        "--target-update",
        "5",
        "--eval-interval",
        "5",
        "--eval-episodes",
        "2",
        "--epsilon-decay",
        "10",
        "--packet-loss",
        "0.3",
        "--sight-range",
        "-1",
        "--reconstruction-weight",
        "0.1",
        "--output-dir",
        str(output_dir),
    ]

    child_env = os.environ.copy()
    if gpu is not None:
        child_env["CUDA_VISIBLE_DEVICES"] = gpu

    run_and_tee(
        command=command,
        cwd=project_root,
        env=child_env,
        log_path=log_path,
    )
    validate_smoke_output(output_dir, log_path)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()

    print("=" * 72)
    print("MATCH CN6 setup and validation")
    print(f"Project root: {project_root}")
    print("=" * 72)

    require_project(project_root)
    paths = ensure_cn6_files(project_root)
    make_backup(project_root, paths)

    for path in paths:
        repair_entry_point(path)

    static_checks(project_root, paths)

    if not args.static_only:
        runtime_dimension_check(project_root)

    if args.static_only or args.skip_smoke:
        print("[done] CN6 repair and checks completed; smoke training skipped")
        return 0

    run_smoke(
        project_root=project_root,
        gpu=args.gpu,
        requested_output=args.smoke_output,
    )
    print("[done] CN6 is ready for the next training stage")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[stopped] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(f"\n[failed] {exc}", file=sys.stderr)
        raise SystemExit(1)
