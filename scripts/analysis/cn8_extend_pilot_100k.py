#!/usr/bin/env python3
"""Resume the four stable CN8 pilot checkpoints from 30k to 100k."""

import argparse
import concurrent.futures
import csv
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time


ROOT = Path(__file__).resolve().parent
TASKS = [
    ("match", 1),
    ("match", 4),
    ("db_invisible", 2),
    ("db_invisible", 4),
]
START_EPISODE = 30000
TARGET_EPISODE = 100000


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def checkpoint_episode(path):
    if not path.is_file():
        return 0
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint["episode"])


def select_resume_checkpoint(output_dir):
    candidates = [
        output_dir / "checkpoint_latest.pt",
        output_dir / "checkpoint_final.pt",
    ]
    episodes = [(checkpoint_episode(path), path) for path in candidates]
    episode, path = max(episodes, key=lambda item: item[0])
    if episode < START_EPISODE:
        raise RuntimeError(
            f"{output_dir.name}: expected a >=30k checkpoint, found {episode}."
        )
    if episode > TARGET_EPISODE:
        raise RuntimeError(
            f"{output_dir.name}: checkpoint episode {episode} exceeds 100k."
        )
    return episode, path


def train_command(method, seed, output_dir, resume_path):
    command = [
        sys.executable,
        "-u",
        str(ROOT / "train_match_cn8.py"),
        "--resume",
        str(resume_path),
        "--episodes",
        str(TARGET_EPISODE),
        "--seed",
        str(seed),
        "--sight-range",
        "0.8",
        "--packet-loss",
        "0.3",
        "--target-update",
        "500",
        "--eval-interval",
        "100",
        "--eval-episodes",
        "20",
        "--learning-rate",
        "1e-4",
        "--grad-clip",
        "5.0",
        "--td-loss",
        "huber",
        "--huber-delta",
        "1.0",
        "--mixer-state-norm",
        "--divergence-threshold",
        "1e5",
        "--gradient-abort-threshold",
        "1e6",
        "--output-dir",
        str(output_dir),
    ]
    if method == "db_invisible":
        command += [
            "--use-local-belief",
            "--belief-hidden-dim",
            "32",
            "--belief-loss-weight",
            "0.01",
            "--belief-supervision",
            "invisible",
        ]
    return command


def evaluation_command(method, checkpoint, episodes, output_dir):
    return [
        sys.executable,
        "-u",
        str(ROOT / "evaluate_cn8_checkpoint.py"),
        "--method",
        method,
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        str(episodes),
        "--expected-checkpoint-episode",
        str(TARGET_EPISODE),
        "--device",
        "cuda",
        "--output-dir",
        str(output_dir),
    ]


def validate_pilot(root):
    plan_path = root / "plan.json"
    if not plan_path.is_file():
        raise SystemExit(f"Missing pilot plan: {plan_path}")
    plan = json.loads(plan_path.read_text())
    planned_tasks = set(plan.get("tasks", []))
    required_tasks = {f"{method}:{seed}" for method, seed in TASKS}
    if not required_tasks.issubset(planned_tasks):
        raise SystemExit("The run directory is not the required four-task pilot.")
    if int(plan.get("episodes", 0)) != START_EPISODE:
        raise SystemExit("The source pilot must end at 30000 episodes.")
    config = plan.get("stable_config", {})
    required_config = {
        "learning_rate": 1e-4,
        "td_loss": "huber",
        "target_update": 500,
        "mixer_state_norm": True,
    }
    for key, expected in required_config.items():
        if config.get(key) != expected:
            raise SystemExit(
                f"Pilot configuration mismatch: {key}={config.get(key)!r}."
            )


def run_extension(args):
    root = args.run_root.resolve()
    validate_pilot(root)
    extension_plan = {
        "source_run_root": str(root),
        "tasks": [f"{method}:{seed}" for method, seed in TASKS],
        "source_episode": START_EPISODE,
        "target_episode": TARGET_EPISODE,
        "evaluation_episodes": args.eval_episodes,
        "gpus": args.gpus,
        "workers": args.workers,
    }
    plan_path = root / "extension_100k_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != extension_plan:
        raise SystemExit(
            "The existing 100k extension has different settings. "
            "Use the same --gpus/--workers/--eval-episodes values."
        )
    write_json(plan_path, extension_plan)

    gpus = args.gpus.split(",")
    if any(not gpu.isdigit() for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise SystemExit("--gpus must contain unique numeric IDs, e.g. 0 or 0,1")

    lock_path = root / "extension_100k.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Another 100k extension launcher is already running.")

        stop_path = root / "STOP_EXTENSION_100K"
        stop_path.unlink(missing_ok=True)
        stopped = threading.Event()
        jobs = list(TASKS)
        mutex = threading.Lock()
        environment_base = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            SDL_VIDEODRIVER="dummy",
            PYGAME_HIDE_SUPPORT_PROMPT="1",
        )
        environment_base["PYTHONPATH"] = (
            str(ROOT) + os.pathsep + environment_base.get("PYTHONPATH", "")
        )

        for gpu in gpus:
            environment = dict(environment_base, CUDA_VISIBLE_DEVICES=gpu)
            subprocess.run(
                [sys.executable, str(ROOT / "cn8_stable.py"), "check"],
                cwd=ROOT,
                env=environment,
                check=True,
            )

        def worker(rank):
            failures = 0
            gpu = gpus[rank % len(gpus)]
            environment = dict(environment_base, CUDA_VISIBLE_DEVICES=gpu)
            while not stopped.is_set():
                with mutex:
                    if not jobs or stop_path.exists():
                        break
                    method, seed = jobs.pop(0)

                name = f"{method}_seed{seed}"
                output_dir = root / "train" / name
                evaluation_dir = root / "eval_100k" / name
                log_path = root / "extend_logs_100k" / f"{name}.log"
                status_path = root / "extend_status_100k" / f"{name}.json"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                status = {
                    "method": method,
                    "seed": seed,
                    "gpu": gpu,
                    "status": "running",
                    "started": dt.datetime.now(dt.timezone.utc).isoformat(),
                }
                write_json(status_path, status)

                def launch(command, phase):
                    status["phase"] = phase
                    write_json(status_path, status)
                    print(f"{phase.upper()} {name} GPU={gpu} log={log_path}", flush=True)
                    with log_path.open("a") as log:
                        log.write("\n" + dt.datetime.now().isoformat() + " " + repr(command) + "\n")
                        log.flush()
                        process = subprocess.Popen(
                            command,
                            cwd=ROOT,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        )
                        status["pid"] = process.pid
                        write_json(status_path, status)
                        while process.poll() is None:
                            if stopped.is_set() or stop_path.exists():
                                stopped.set()
                                process.terminate()
                                try:
                                    process.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    process.kill()
                                    process.wait()
                                raise InterruptedError("Stopped; rerun the same command to resume.")
                            time.sleep(0.5)
                        if process.returncode:
                            raise RuntimeError(
                                f"{phase} exited with code {process.returncode}; see {log_path}"
                            )

                try:
                    final_checkpoint = output_dir / "checkpoint_final.pt"
                    backup_checkpoint = output_dir / "checkpoint_30000.pt"
                    final_episode = checkpoint_episode(final_checkpoint)
                    if final_episode == START_EPISODE and not backup_checkpoint.exists():
                        shutil.copy2(final_checkpoint, backup_checkpoint)

                    current_episode, resume_path = select_resume_checkpoint(output_dir)
                    if current_episode < TARGET_EPISODE:
                        launch(
                            train_command(method, seed, output_dir, resume_path),
                            "train",
                        )

                    if checkpoint_episode(final_checkpoint) != TARGET_EPISODE:
                        raise RuntimeError("Missing final 100k checkpoint.")

                    summary_path = evaluation_dir / "evaluation_summary.json"
                    evaluation_ready = False
                    if summary_path.is_file():
                        summary = json.loads(summary_path.read_text())
                        evaluation_ready = (
                            int(summary.get("checkpoint_episode", 0)) == TARGET_EPISODE
                            and int(summary.get("evaluation_episodes", 0)) == args.eval_episodes
                        )
                    if not evaluation_ready:
                        launch(
                            evaluation_command(
                                method,
                                final_checkpoint,
                                args.eval_episodes,
                                evaluation_dir,
                            ),
                            "eval",
                        )

                    status["status"] = "done"
                    print("PASS", name, flush=True)
                except InterruptedError as error:
                    status.update(status="stopped", error=str(error))
                    failures += 1
                except Exception as error:
                    status.update(status="failed", error=str(error))
                    print("FAIL", name, str(error), flush=True)
                    failures += 1
                finally:
                    status.pop("pid", None)
                    status["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
                    write_json(status_path, status)
            return failures

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker, rank) for rank in range(args.workers)]
            try:
                failures = sum(future.result() for future in futures)
            except KeyboardInterrupt:
                stopped.set()
                failures = sum(future.result() for future in futures) + 1
        if stop_path.exists() or stopped.is_set():
            failures += 1
        print("Run directory:", root, "| failures:", failures, flush=True)
        return int(bool(failures))


def latest_logged_episode(path):
    if not path.is_file():
        return "?"
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 8192))
        lines = handle.read().decode(errors="replace").splitlines()
    for line in reversed(lines):
        first = line.split(",", 1)[0]
        if first.isdigit():
            return first
    return "?"


def watch(root):
    root = root.resolve()
    print(dt.datetime.now().isoformat(timespec="seconds"))
    print(f"{'TASK':24} {'EPISODE':18} {'STATUS':12} PHASE")
    completed = 0
    for method, seed in TASKS:
        name = f"{method}_seed{seed}"
        status_path = root / "extend_status_100k" / f"{name}.json"
        status = json.loads(status_path.read_text()) if status_path.is_file() else {"status": "waiting"}
        shown_status = status["status"]
        if shown_status == "running" and status.get("pid"):
            try:
                os.kill(status["pid"], 0)
            except ProcessLookupError:
                shown_status = "interrupted"
        completed += shown_status == "done"
        episode = latest_logged_episode(root / "train" / name / "training_log.csv")
        print(
            f"{name:24} {episode + '/' + str(TARGET_EPISODE):18} "
            f"{shown_status:12} {status.get('phase', '')}"
        )
    print(f"Completed {completed}/{len(TASKS)}")


def pack(root, include_checkpoints=False):
    root = root.resolve()
    output = root.parent / (
        "cn8_stability_pilot_100k_results_"
        + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".tar.gz"
    )
    allowed_suffixes = {".csv", ".json", ".log", ".pdf", ".png"}
    with tarfile.open(output, "w:gz") as archive:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix in allowed_suffixes or (
                include_checkpoints and path.suffix == ".pt"
            ):
                archive.add(path, arcname=str(path.relative_to(root.parent)))
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["run", "watch", "stop", "pack"])
    parser.add_argument(
        "--run-root",
        type=Path,
        default=ROOT / "runs" / "cn8_stability_pilot_30k",
    )
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=500)
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.eval_episodes < 1:
        parser.error("workers and eval-episodes must be positive")

    if args.mode == "run":
        return run_extension(args)
    if args.mode == "watch":
        watch(args.run_root)
    elif args.mode == "stop":
        stop_path = args.run_root.resolve() / "STOP_EXTENSION_100K"
        stop_path.touch()
        print("Stop requested. Rerun the same run command to resume safely.")
    else:
        pack(args.run_root, args.include_checkpoints)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
