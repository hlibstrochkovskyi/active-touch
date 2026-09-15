# Touch Explorer

A simulated probe learns a fixed object's 2D outline from noisy contacts. It uses a periodic Gaussian process to estimate radius and uncertainty, then chooses where to probe next.

The working core includes five shapes, five policies, exact motion costs, reproducible logs, comparison plots, and an offline browser replay. The object must be star-shaped about the known reference point: one boundary crossing per radial approach.

## Run

The project uses Python 3.12, managed by `uv`.

```bash
uv sync --locked
uv run --frozen touch-explorer demo --config configs/demo.toml --output results/demo
xdg-open results/demo/replay.html
```

The replay has play/pause, speed, scrubbing, and a truth toggle. It reads saved posterior snapshots; playback does not fit models. The output also includes `summary.png`, `summary.svg`, and raw measurements.

Choose a fresh output directory for each run. Existing runs are preserved.

## Compare policies

```bash
uv run --frozen touch-explorer benchmark --config configs/pilot.toml --seeds 2 --output results/pilot
```

This development pilot runs five policies on two instances of each of five shape families: 50 episodes. The CLI generates the shapes; the `[shape]` section is used by `demo`. Add `--shapes circle,ellipse` for a smaller comparison.

| Policy | Selection rule |
|---|---|
| `random` | Unvisited random angle |
| `sweep` | Clockwise uniform scan |
| `gap` | Midpoint of the largest angular gap |
| `variance` | Largest GP variance |
| `cost` | Expected average variance reduction / estimated motion time |

Results include `comparison.png`, `summary.json`, and each episode's observations and metrics. Failed runs remain in `runs.json`; the command returns a nonzero status if any fail. Time comparisons hold estimates constant between completed cycles.

The [first pilot](docs/first-pilot.md) completed all 50 episodes. The simple gap baseline was slightly better than the cost policy in this small sample.

## Tests

```bash
uv run --frozen pytest
uv run --frozen ruff check src tests
uv run --frozen ruff format --check src tests
```

Tests cover analytic geometry, collision/noise separation, motion accounting, the GP posterior, variance-reduction math, shared initial measurements, data isolation, reproducibility, CLI output, and failure reporting.

## Saved data

- `config.json`, `metadata.json`: resolved settings, seeds, versions, and source revision.
- `observations.jsonl`: noisy sensor readings only.
- `decisions.jsonl`: chosen actions, scores, and estimated costs.
- `truth.jsonl`: exact geometry and events for evaluation/replay only.
- `metrics.csv`: errors, interval coverage, motion time, and computation time.
- `snapshots.npz`: prior and posterior curves after each completed cycle.

Rebuild a run's figures and replay without executing the simulator:

```bash
uv run --frozen touch-explorer replay results/demo
```

## Next milestones

The current model has fixed hyperparameters and always runs to its touch budget. Online parameter fitting, coverage safeguards, stopping diagnostics, and the final held-out study remain planned. General implicit surfaces and hardware are later extensions.

See the [project plan](PROJECT_PLAN.md) for the mathematics and milestones, and [research notes](RESEARCH_NOTES.md) for sources and assumptions. Everything runs on the CPU; no pretrained model or external dataset is needed.
