# Development reliability study — 2026-09-16

60 runs completed without numerical failures: 12 matched cases across five variants. The guarded stopping rule still missed narrow recesses. This is a development study, not the final held-out evaluation.

## Reproduce

```bash
uv run --frozen touch-explorer reliability --config configs/reliability.toml --seeds 4 --output results/reliability
```

The recorded implementation revision is `695d4f8`; the working tree was clean when the study ran. Raw outputs are in `results/reliability-2026-09-16/`, excluded from Git. Settings and library versions are recorded in `manifest.json`.

- Four generated instances each of circle, ellipse, and narrow recess, with one noise realization per instance.
- Maximum 128 touches, including eight shared initialization touches.
- 256 candidate angles, 128 variance-integration points, 2048 truth-only evaluation points.
- Gaussian sensor noise standard deviation 0.005; exact angle and immobilized object.
- Learned fits at touch 16 and every eight touches thereafter, retaining the fitted parameters in between.
- Coverage variant: largest-gap probing every fifth post-initialization action.
- Both stopping variants use that coverage schedule. The guarded variant additionally requires 24 touches, a maximum gap of 15 degrees, and two new audit touches.
- A confidence stop is false when radial RMSE exceeds 0.01 or sampled maximum radial error exceeds 0.03. These thresholds were set before running the study.

## Results

| Variant | Mean touches | Mean radial RMSE | Cases meeting both error limits | False confidence stops |
|---|---:|---:|---:|---:|
| Fixed parameters, full budget | 128.00 | 0.004220 | 12/12 | No confidence stops |
| Learned parameters, full budget | 128.00 | 0.002690 | 11/12 | No confidence stops |
| Learned + coverage, full budget | 128.00 | 0.002685 | 12/12 | No confidence stops |
| Uncertainty stop | 23.92 | 0.008277 | 9/12 | 3/12 |
| Guarded stop | 38.75 | 0.006295 | 10/12 | 2/12 |

Every stopping run terminated by its confidence rule before exhausting the budget. No optimizer fallbacks occurred. Full-budget variants used substantially more measurements, so their smaller errors do not by themselves establish a more efficient policy.

The false stops all occurred in the recess family. In case `recess-4002012`, uncertainty-only stopping ended at 16 touches with maximum radial error 0.2144. Guarded stopping ended at 34 touches and still had maximum error 0.2057. The additional probes did not resolve the recess.

Guarded stopping corrected the false stop in `recess-4002013`, but `recess-4002015` still failed the thresholds at 34 touches. At the full 128-touch budget, the learned-only variant also narrowly missed the maximum-error target on `recess-4002012` (0.0314).

Mean pointwise interval coverage was approximately 95.7% for uncertainty stopping and 94.7% for guarded stopping. Those averages can look satisfactory while a small part of an outline is badly wrong. They are not simultaneous whole-outline guarantees.

## Inspect the failure

```bash
uv run --frozen touch-explorer replay results/reliability-2026-09-16/recess-4002012-guarded-stop
xdg-open results/reliability-2026-09-16/recess-4002012-guarded-stop/replay.html
```

The replay uses saved posterior snapshots. `model.jsonl` records parameter values and optimizer outcomes; `decisions.jsonl` identifies coverage and audit touches. The end label reports the actual confidence stop, rather than presenting it as a guaranteed reconstruction.

## What this supports

These cases show that fitting improves average reconstruction at a full budget, while a small uncertainty band can still overlook local geometry. Coverage and audits helped one additional case here, but 12 cases and one noise realization each are too few to estimate a stable failure rate. Neither 25% nor 16.7% should be generalized to other shapes or noise levels.

Keep these cases as development examples. Next evaluate separate noise seeds, sensor-model mismatch, and a held-out shape set with paired uncertainty intervals. Do not tune stopping thresholds on these results and call the same cases a final validation.
