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

## Test confidence and stopping

```bash
uv run --frozen touch-explorer reliability --config configs/reliability.toml --seeds 4 --output results/reliability
```

This runs 60 matched development episodes: circle, ellipse, and narrow-recess cases across five variants. They isolate fixed versus learned GP parameters, a largest-gap touch every fifth post-initialization action, and uncertainty-only versus guarded stopping. Both stopping variants use the same coverage safeguard.

Guarded stopping requires the configured interval width, minimum touch count, angular coverage, and two additional audit touches with acceptable prediction residuals. A confidence stop is a model decision, not a correctness guarantee. The evaluator separately flags a false stop when final radial RMSE exceeds 0.01 or sampled maximum error exceeds 0.03.

`reliability.json` reports stops, false stops, exhausted budgets, errors, interval coverage, and optimizer fallbacks. No-stop cases have a null false-stop rate. `reliability.png` compares the variants; these are development results with different actual sensing budgets.

The [first reliability study](docs/reliability-study.md) completed 60 runs. Uncertainty-only stopping failed the accuracy criteria in 3/12 cases; guarded stopping in 2/12. The remaining failures expose narrow recesses that the model confidently misses.

For individual runs, `[model] learn_hyperparameters = true` fits bounded kernel parameters at touch 16 and every 8 touches afterward. Learned values persist between optimizations and warm-start the next fit. A failed optimization refits all current data with the last valid parameters and records the failure. `[experiment] safeguard_every = 5` enables periodic gap probing; `[stopping] mode` accepts `budget` (default), `uncertainty`, or `guarded`.

## Test sensor mismatch

```bash
uv run --frozen touch-explorer mismatch --config configs/mismatch.toml --seeds 2 --output results/mismatch
```

This runs 48 development episodes: four sensor conditions, two methods, three shape families, and two instances per family. Conditions are nominal sensing, a constant +0.015 radius bias, three times the assumed noise standard deviation, and 5% additive Gaussian contamination with standard deviation 0.05. Each fault is tested separately. Both methods learn GP parameters and probe the largest gap every fifth later action; one uses the full budget, the other guarded stopping. Settings and accuracy thresholds are fixed before running the comparison.

`mismatch.json`, `mismatch.png`, and `mismatch.svg` report the same metrics as the reliability study. The manifest records every resolved condition. The command replaces `[sensor]` with the four prescribed conditions; `demo`, `benchmark`, and `reliability` instead use the configured sensor settings.

For a custom sensor in those commands:

```toml
[sensor]
noise_scale = 3.0
bias = 0.0
outlier_probability = 0.0
outlier_std = 0.0
```

`[experiment] noise_std` remains the agent's assumed noise standard deviation. The simulator multiplies it by `noise_scale`, adds `bias`, and independently adds a zero-mean Gaussian outlier with the specified probability and standard deviation. Readings are not clipped. Fault settings are private to the simulator; observations still report the assumed variance. Geometry and physical motion remain exact. Default sensor settings reproduce earlier measurements exactly.

Repeated touches cannot identify a constant sensor offset separately from object radius without additional calibration information. These experiments measure degradation; the GP has no bias correction or robust outlier likelihood.

The [sensor mismatch study](docs/mismatch-study.md) completed all 48 episodes. Guarded stopping failed the accuracy criteria in 2/6 nominal cases, 6/6 biased cases, and 5/6 outlier cases. With underestimated noise, all three confidence stops were false and the other three runs exhausted the budget.

## Held-out primary benchmark

```bash
uv run --frozen touch-explorer heldout --protocol configs/heldout.json --workers 2 --output results/heldout
uv run --frozen touch-explorer analyze results/heldout
```

The [frozen protocol](docs/heldout-protocol.md) specifies 600 episodes on 20 new shapes, three correctly specified noise levels, two independent sensor-noise seeds, and five policies. It uses fixed GP parameters and 128 touches. Shape parameters, seeds, and analysis choices are stored before execution.

`analysis.json` reports family/noise-level means and paired differences against largest-gap sampling. Bootstrap intervals resample whole shapes within families after averaging noise repeats. A missing or failed run withholds paired inference. `comparison.png` shows error against touches and modeled time; all raw episode logs remain replayable. The `analyze` command regenerates these outputs without rerunning the simulator.

The [held-out study](docs/heldout-study.md) completed all 600 episodes. Cost-aware probing showed no clear advantage over largest-gap sampling on the predeclared 50-second metric. The report includes paired intervals, all run endpoints, family-level results, and a denser-grid check of the recess cases.

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
- `model.jsonl`: fitted parameters, optimization attempts, warnings, and fallback reasons.
- `snapshots.npz`: prior and posterior curves after each completed cycle.

Rebuild a run's figures and replay without executing the simulator:

```bash
uv run --frozen touch-explorer replay results/demo
```

## Next milestones

The primary held-out benchmark and paired analysis are complete. Remaining work is held-out evaluation of learning/stopping and the final report across studies. General implicit surfaces and hardware are optional later extensions.

See the [project plan](PROJECT_PLAN.md) for the mathematics and milestones, and [research notes](RESEARCH_NOTES.md) for sources and assumptions. Everything runs on the CPU; no pretrained model or external dataset is needed.
