# Frozen learning and stopping evaluation

This follow-up uses **fresh cases**, not the 20 shapes whose primary results have already been inspected. Methods and thresholds remain those used in development. All settings and generated shapes are stored in [heldout-reliability.json](../configs/heldout-reliability.json) before execution.

## Matrix

Four circles, four ellipses, and four narrow recesses, each with two independent noise realizations and all five development variants: **12 × 2 × 5 = 120 episodes**.

Shape seeds are `1700000 + 10000*family_index + instance_index`, with family indices from the existing generator order (circle 0, ellipse 1, recess 4). Episode seeds use 1800000 in place of 1700000. Noise seed labels are 52001 and 52002, independently derived per episode. Candidate phase and policy randomness are unchanged across noise repeats; candidate/repeat noise is matched across variants.

Use correctly specified Gaussian noise with standard deviation 0.005, eight initial touches, a cap of 128 touches, 256 candidates, 256 integration points, and 4096 truth-only evaluation angles. Exact motion and shape data remain inaccessible to the controller. This is a nominal-sensor evaluation; the sensor-mismatch study remains separate.

| Variant | Learning | Gap safeguard | Stop rule |
|---|---|---|---|
| fixed-budget | Off | Off | 128 touches |
| learned-budget | On | Off | 128 touches |
| learned-coverage | On | Every fifth post-initialization action | 128 touches |
| uncertainty-stop | On | Same safeguard | Maximum pointwise half-width ≤ 0.02 |
| guarded-stop | On | Same safeguard | Width plus coverage and new audits |

All variants use the cost acquisition policy, matching the development ablation. This isolates the existing learning and stopping mechanisms; it does not retest policy rankings or claim they transfer to gap acquisition.

The initial GP has mean 0.525, signal standard deviation 0.2, and length scale 0.4. Learning fits the signal and length scale at touch 16 and every eight touches thereafter, with bounds [0.05, 0.4] and [0.08, 1.2]. Guarded stopping retains minimum 24 touches, maximum angular gap 15 degrees, two new gap audits, and absolute normalized innovation ≤ 2.5. No thresholds are adjusted after seeing results.

## Questions and analysis

Predeclare these paired contrasts:

1. `learned-budget − fixed-budget`: final RMSE, isolating fitting at equal budgets.
2. `learned-coverage − learned-budget`: final RMSE, isolating coverage at equal budgets.
3. `guarded-stop − uncertainty-stop`: false-confidence event frequency, accuracy success, touches, modeled motion time, and final RMSE. These describe a tradeoff under different actual budgets.

A confidence stop is false when RMSE exceeds 0.01 or sampled maximum radial error exceeds 0.03. Report confidence-stop counts, false stops **among confidence stops**, exhausted budgets, and accuracy successes separately. The paired false-stop contrast uses a per-episode event indicator across all scheduled episodes; a controller that never stops has zero such events but is not thereby reliable. No-stop conditional rates are null, and budget exhaustion is always shown.

Average the two noise repeats within each shape, then average within families with equal family weights. Nominal 95% percentile bootstrap intervals use 10,000 resamples of whole shapes within families, seed 92001. This gives 12 shape-level units, not 24 independent observations. Intervals are approximate with four shapes per family and are not multiplicity-adjusted. Conditional false-stop rates are descriptive counts, without treating the paired event interval as their interval.

For plots and the secondary 50-second time average, hold a stopped episode's final estimate after stopping. Padded touch curves represent available budgets, not additional measurements. Actual touches and motion are recorded separately. Missing or failed runs withhold paired inference.

The same eight initial contacts must match across all five variants within each shape/noise pair. Preserve configurations, source revision, model fit diagnostics, measurements, and snapshots. Choose illustrative failures only after aggregate analysis and label that selection as descriptive.

## Run

```bash
uv run --frozen touch-explorer heldout --protocol configs/heldout-reliability.json --workers 2 --output results/heldout-reliability
uv run --frozen touch-explorer analyze results/heldout-reliability
```

The manifest freezes the protocol hash and source revision. The earlier primary protocol and its archived results retain their original meaning.
