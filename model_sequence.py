#!/usr/bin/env python3
"""Run experiment servers sequentially, with manual handoffs (Linux/macOS)."""

import argparse
import os
from pathlib import Path
import select
import shlex
import signal
import subprocess
import sys
import time


# name, directory, command, environment overrides
MODELS = [
    ("Laya", "laya", ["laya-serve"],
     {"LAYA_HOST": "0.0.0.0", "LAYA_PORT": "8010", "LAYA_DEVICE": "cuda", "LAYA_PRELOAD": "0"}),
    ("nimble", "nimble", ["python", "nimble_server.py"], {}),
    ("semif", "SemIf-OpenJev", ["python", "semif_server.py"], {}),
    ("Rizzo-flow", "rizzo-flow", ["bash", "start_rizzo.sh"], {}),
    ("Von", "von", ["bash", "start_von.sh"], {}),
    ("NanoJev", "NanoJev", ["bash", "start_nanojev.sh"], {}),
    ("LiquidAI_d1_3B", "d1-server", ["python", "d1_server.py"], {"CUDA_VISIBLE_DEVICES": "0"}),
    ("Decider", "decider-server",
     ["python", "-m", "uvicorn", "decider.serve:app", "--host", "0.0.0.0",
      "--port", "8018", "--workers", "1"],
     {"CUDA_VISIBLE_DEVICES": "0", "DECIDER_MODEL": "Mapika/decider-2b",
      "DECIDER_DEVICE": "cuda", "DECIDER_IMAGES": "0"}),
    ("h2o_lightning_4B", "h2o-lightning-server", ["bash", "start_h2o_lightning.sh"], {}),
    ("Neohorse", "neohorse-server", ["bash", "start_neohorse.sh"], {}),
    ("d3-flash", "d3-flash-server", ["bash", "start_d3_flash.sh"], {}),
]


def group_exists(process):
    process.poll()  # Reap the leader if it exited.
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False


def stop_server(process, timeout):
    """Stop the session's process group, including shell-script children."""
    for sig, wait in ((signal.SIGTERM, timeout), (signal.SIGKILL, 5)):
        if not group_exists(process):
            return
        print(f"Stopping server processes ({sig.name})...", flush=True)
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + wait
        while group_exists(process) and time.monotonic() < deadline:
            time.sleep(0.1)
    if group_exists(process):
        raise RuntimeError("Server process group still exists; refusing to start another model.")


def launch_spec(base_dir, model):
    name, directory, command, overrides = model
    folder = base_dir / directory
    venv = folder / ".venv"
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing model directory: {folder}")
    if not (venv / "bin" / "python").is_file():
        raise FileNotFoundError(f"Missing virtual environment: {venv}")
    command = list(command)
    if command[0] in ("python", "laya-serve"):
        command[0] = str(venv / "bin" / command[0])
        if not os.access(command[0], os.X_OK):
            raise FileNotFoundError(f"Missing executable: {command[0]}")
    if len(command) > 1 and command[1].endswith((".py", ".sh")):
        if not (folder / command[1]).is_file():
            raise FileNotFoundError(f"Missing server script: {folder / command[1]}")
    env = os.environ.copy()
    env.pop("PYTHONHOME", None)
    env.update(VIRTUAL_ENV=str(venv), PATH=f"{venv / 'bin'}:{env.get('PATH', '')}",
               PYTHONUNBUFFERED="1")
    env.update(overrides)
    return folder, command, env


def run_model(base_dir, model, log_dir, timeout):
    folder, command, env = launch_spec(base_dir, model)
    log_path = log_dir / f"{model[1]}.log"
    print(f"\nStarting {model[0]}: {shlex.join(command)}")
    print(f"Logs: tail -f {shlex.quote(str(log_path))}")
    with log_path.open("a") as log:
        log.write(f"\n--- Started {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        log.flush()
        process = subprocess.Popen(command, cwd=folder, env=env,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            print("Check the logs for readiness, then run your experiments in another terminal.")
            print("When finished, press Enter to stop this model (or q + Enter to quit).", flush=True)
            while process.poll() is None:
                if select.select([sys.stdin], [], [], 0.5)[0]:
                    answer = sys.stdin.readline()
                    return bool(answer) and answer.strip().lower() not in ("q", "quit", "exit")
            print(f"Server launcher exited with code {process.returncode}. Check {log_path}.")
            return True
        finally:
            # A second Ctrl+C must not bypass cleanup and leave GPU workers behind.
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                stop_server(process, timeout)
            finally:
                signal.signal(signal.SIGINT, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Parent of laya/, nimble/, etc. (default: this script's directory)")
    parser.add_argument("--start-at", choices=[m[0] for m in MODELS], default=MODELS[0][0],
                        help="Skip earlier models and start here")
    parser.add_argument("--log-dir", type=Path, help="Default: BASE_DIR/model-logs")
    parser.add_argument("--stop-timeout", type=float, default=30,
                        help="Seconds before force-killing server processes (default: 30)")
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("This launcher requires Linux or macOS process groups.")
    if not sys.stdin.isatty():
        parser.error("Run this script in an interactive terminal.")
    if not 0 < args.stop_timeout < float("inf"):
        parser.error("--stop-timeout must be a finite positive number.")
    base_dir = args.base_dir.expanduser().resolve()
    log_dir = (args.log_dir or base_dir / "model-logs").expanduser().resolve()
    models = MODELS[next(i for i, m in enumerate(MODELS) if m[0] == args.start_at):]
    log_dir.mkdir(parents=True, exist_ok=True)
    for i, model in enumerate(models):
        if i:
            answer = input(f"\nDeploy {model[0]} next? [y/N] ").strip().lower()
            if answer not in ("y", "yes"):
                break
        try:
            if not run_model(base_dir, model, log_dir, args.stop_timeout):
                break
        except (FileNotFoundError, PermissionError) as exc:
            print(f"Cannot start {model[0]}: {exc}", file=sys.stderr)
    print("Sequence finished.")


def interrupted(signum, frame):
    raise KeyboardInterrupt


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped.")
    except (OSError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
