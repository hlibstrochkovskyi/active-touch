"""Run a demo, replay saved data, or compare sensing policies and stopping rules."""

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import Config, load_config
from .heldout import analyze_folder, run_heldout
from .plotting import comparison, render_run
from .policies import POLICIES
from .reliability import run_study
from .runner import provenance, run_episode, save_episode, write_json
from .world import Shape

SHAPES = ("circle", "ellipse", "rectangle", "fourier", "recess")


def benchmark_shape(kind: str, seed: int) -> Shape:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 67]))
    parameters = {"kind": kind, "phase": float(rng.uniform(0, 2 * np.pi))}
    if kind == "circle":
        parameters["base"] = float(rng.uniform(0.45, 0.65))
    elif kind == "ellipse":
        parameters.update(a=float(rng.uniform(0.55, 0.72)), b=float(rng.uniform(0.3, 0.48)))
    elif kind == "rectangle":
        parameters.update(a=float(rng.uniform(0.45, 0.6)), b=float(rng.uniform(0.28, 0.4)))
    elif kind == "fourier":
        parameters["base"] = float(rng.uniform(0.48, 0.65))
    else:
        parameters.update(
            base=float(rng.uniform(0.55, 0.7)),
            depth=float(rng.uniform(0.1, 0.25)),
            width=float(rng.uniform(0.04, 0.12)),
        )
    return Shape(**parameters)


def benchmark(config: Config, folder: Path, seeds: int, shapes: list[str]) -> int:
    if seeds < 1 or not shapes or len(set(shapes)) != len(shapes) or set(shapes) - set(SHAPES):
        raise ValueError(
            "use positive --seeds and a comma-separated list of supported unique shapes"
        )
    folder.mkdir(parents=True, exist_ok=False)
    write_json(
        folder / "manifest.json",
        {
            "config": asdict(config),
            "seeds_per_shape": seeds,
            "shapes": shapes,
            "policies": POLICIES,
            "scope": "development pilot",
            **provenance(),
        },
    )
    rows = []
    for kind in shapes:
        for repeat in range(seeds):
            seed = config.experiment.seed + SHAPES.index(kind) * 1_000_003 + repeat
            shape = benchmark_shape(kind, seed)
            case = f"{kind}-{seed}"
            for policy in POLICIES:
                run_name = f"{case}-{policy}"
                run_config = replace(
                    config,
                    shape=shape,
                    experiment=replace(config.experiment, seed=seed, policy=policy),
                )
                row = {"case": case, "run": run_name, "policy": policy}
                start = perf_counter()
                try:
                    result = run_episode(run_config)
                    save_episode(result, folder / run_name)
                    row.update(
                        status=result.status,
                        final_rmse=result.metrics[-1]["rmse"],
                        actual_touches=len(result.observations),
                    )
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    failure_folder = folder / run_name
                    failure_folder.mkdir(exist_ok=True)
                    write_json(
                        failure_folder / "failure.json", {"config": asdict(run_config), **row}
                    )
                row["wall_seconds"] = perf_counter() - start
                rows.append(row)
                write_json(folder / "runs.json", rows)
                print(f"{run_name}: {row['status']} ({row['wall_seconds']:.2f}s)", flush=True)
    comparison(folder, rows)
    failures = sum(row["status"] == "failed" for row in rows)
    print(f"Saved {len(rows)} runs; {failures} failed: {folder.resolve()}")
    return 1 if failures else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Contact-based shape exploration")
    commands = parser.add_subparsers(dest="command", required=True)
    heldout = commands.add_parser("heldout", help="run a frozen primary benchmark protocol")
    heldout.add_argument("--protocol", type=Path, required=True)
    heldout.add_argument("--output", type=Path, default=Path("results/heldout"))
    heldout.add_argument("--workers", type=int, default=1)
    analysis = commands.add_parser("analyze", help="rebuild held-out analysis without simulation")
    analysis.add_argument("folder", type=Path)
    for name in ("demo", "benchmark", "reliability", "mismatch"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path)
        command.add_argument("--output", type=Path, default=Path("results") / name)
        if name in ("benchmark", "reliability", "mismatch"):
            command.add_argument("--seeds", type=int, default=2)
            command.add_argument(
                "--shapes",
                default=",".join(SHAPES) if name == "benchmark" else "circle,ellipse,recess",
            )
    replay = commands.add_parser("replay", help="render saved snapshots without fitting")
    replay.add_argument("folder", type=Path)
    try:
        args = parser.parse_args(argv)
        if args.command == "heldout":
            return run_heldout(json.loads(args.protocol.read_text()), args.output, args.workers)
        if args.command == "analyze":
            return int(not analyze_folder(args.folder)["complete"])
        if args.command == "replay":
            render_run(args.folder)
            print(f"Replay: {(args.folder / 'replay.html').resolve()}")
            return 0
        config = load_config(args.config) if args.config else Config()
        if args.command in ("reliability", "mismatch"):
            return run_study(
                config,
                args.output,
                args.seeds,
                args.shapes.split(","),
                benchmark_shape,
                SHAPES,
                study=args.command,
            )
        if args.command == "benchmark":
            return benchmark(config, args.output, args.seeds, args.shapes.split(","))
        if args.output.exists():
            raise FileExistsError(f"{args.output} already exists; choose a new --output")
        start = perf_counter()
        result = run_episode(config)
        save_episode(result, args.output)
        render_run(args.output)
        print(
            f"{len(result.observations)} / {config.experiment.touches} touches · {result.status}"
            f" · RMSE {result.metrics[-1]['rmse']:.4f}"
            f" · {perf_counter() - start:.2f}s wall time"
        )
        print(f"Replay: {(args.output / 'replay.html').resolve()}")
        return 0
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
