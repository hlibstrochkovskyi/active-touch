# Held-out primary results — 2026-09-18

All **600 episodes** completed without execution failures. Cost-aware probing did **not demonstrate an advantage over largest-gap probing** on the predeclared time-efficiency metric. This result supports keeping the simple baseline; it does not establish equivalence on arbitrary shapes.

## Design and provenance

The [protocol](heldout-protocol.md) and all shape parameters were committed at `f2c1bc4` before executing the held-out cases. Every episode recorded that revision with a clean working tree. No parameters or analysis settings were changed after seeing outcomes.

The matrix contains 20 shapes, balanced across five families, three correctly specified Gaussian noise levels, two sensor-noise seeds, and five policies. All policies use fixed GP parameters and complete 128 touches. Initial readings match exactly within each comparison. Inference uses 20 shape-level units, averaging noise repeats within shapes and weighting families equally.

The primary metric is mean radial RMSE over the first 50 modeled seconds, integrating completed-observation estimates exactly. The horizon was selected from development timing. Nominal 95% percentile bootstrap intervals resample shapes within families; they are approximate with only four shapes per family and are not corrected for multiple comparisons. See the [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html) for the percentile construction and preservation of paired observations during resampling. This implementation additionally stratifies by family.

## Main comparison

Lower errors are better. Time averages include the common initialization; total motion time covers all 128 touches and is a separate quantity.

| Policy | Final RMSE | Time-averaged RMSE, 0–50 s | Total modeled motion, s |
|---|---:|---:|---:|
| Random | 0.005344 | 0.039004 | 398.26 |
| Sweep | 0.004775 | 0.040454 | 217.04 |
| Largest gap | 0.004775 | 0.035311 | 238.71 |
| Maximum variance | 0.004775 | 0.037439 | 361.95 |
| Cost-aware | 0.004817 | 0.035289 | 270.54 |

The primary **cost minus gap** difference is **−0.0000226**, with interval **[−0.0003057, +0.0002979]**. The interval includes both improvement and degradation. Final RMSE also shows no clear cost advantage: difference +0.0000419, interval [−0.0000503, +0.0001315].

On the 50-second metric, the other policy-minus-gap differences were:

| Policy | Mean difference | Nominal 95% interval |
|---|---:|---:|
| Random | +0.003693 | [+0.003035, +0.004376] |
| Sweep | +0.005143 | [+0.004178, +0.006233] |
| Maximum variance | +0.002128 | [+0.001574, +0.002780] |

Sweep, gap, and variance collected identical final measurement sets in every one of the 120 matched shape/noise groups. Their final estimates agree to floating-point precision; tiny differences around 1e-17 in the raw analysis are numerical roundoff. Their ordering and motion costs differ substantially. Sweep finishes with the least total motion, but has poorer global reconstruction during the early 50-second horizon.

Gap already breaks equal-gap ties using predicted travel cost. It is therefore a strong baseline for the cost-aware policy, not an arbitrary travel-oblivious sampler.

![Error against touch count and modeled time](data/heldout-comparison.svg)

## Noise and shape breakdown

The cost-minus-gap time-efficiency intervals include zero at every noise level:

| Noise standard deviation | Mean difference | Nominal 95% interval |
|---|---:|---:|
| 0 | +0.0000865 | [−0.0002204, +0.0004594] |
| 0.005 | −0.0001710 | [−0.0004772, +0.0001349] |
| 0.015 | +0.0000166 | [−0.0003545, +0.0003948] |

Descriptive time-averaged errors by family, pooled over noise levels:

| Family | Gap | Cost |
|---|---:|---:|
| Circle | 0.016340 | 0.016168 |
| Ellipse | 0.040362 | 0.040062 |
| Rectangle | 0.054261 | 0.054348 |
| Fourier | 0.027648 | 0.027890 |
| Recess | 0.037946 | 0.037974 |

Both policies improve with more touches. Gap's mean RMSE at 8/16/32/64/128 touches was 0.026382/0.011816/0.008129/0.005977/0.004775; cost's was 0.026382/0.011958/0.008585/0.006331/0.004817. These are descriptive learning curves, not independent tests at every checkpoint.

## Reliability and resolution

The development accuracy limits were RMSE ≤ 0.01 and sampled maximum error ≤ 0.03. At 128 touches, 97/120 runs met both limits for each of gap, sweep, variance, and cost; random met them in 88/120. These dependent run counts are descriptive. All runs stopped at their budget, so none is a confidence-stop claim.

The illustrative cost-policy failure `recess-2-n2-s51001-cost` had RMSE 0.011276 and sampled maximum error 0.032965. Its pointwise interval coverage was 96.4%, yet it missed the accuracy targets. The example was selected after analysis as the cost-policy recess run with largest final maximum error. A successful example, `ellipse-0-n1-s51001-cost`, had RMSE 0.003834 and maximum error 0.013334.

All 120 recess runs were re-evaluated on 8192 angles. Reconstructed predictions first matched saved final snapshots on the original grid. Doubling grid density changed RMSE by at most 6.24e-11 and sampled maximum error by at most 6.25e-6. This supports numerical resolution of these reported metrics, not a continuous whole-outline guarantee.

Execution took about **18.5 minutes with two CPU workers**; the raw replayable data occupies approximately 3.9 GB. A development smoke run at the same settings used about 159 MiB peak resident memory. The mean within-episode 95th-percentile fitting/selection time was 25.1 ms for cost versus 8.0 ms for gap; these timings are descriptive under concurrent execution.

## Inspect or reproduce

- [600 run endpoints and checkpoints](data/heldout-runs.csv)
- [Full aggregate analysis, intervals, and mean curves](data/heldout-analysis.json)
- [Frozen cases and settings](../configs/heldout.json)

```bash
uv run --frozen touch-explorer heldout --protocol configs/heldout.json --workers 2 --output results/heldout
uv run --frozen touch-explorer analyze results/heldout
uv run --frozen python scripts/check_heldout_resolution.py results/heldout
uv run --frozen touch-explorer replay results/heldout/recess-2-n2-s51001-cost
```

The original raw directory is `results/heldout-2026-09-18/`, excluded from Git. Both example replays are available there. This completes the primary fixed-model comparison. Held-out comparisons of learning and stopping, and the final synthesis across studies, remain separate work; the development sensor-mismatch findings are not corrected by this benchmark.
