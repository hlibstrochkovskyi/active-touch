# Frozen primary benchmark protocol

Frozen before inspecting held-out outcomes. The complete machine-readable design is [configs/heldout.json](../configs/heldout.json); each run also saves its hash and source revision. Changes after outcome inspection require a new development study, not relabeling these cases as unseen data.

## Question and design

Does expected variance reduction per modeled motion time improve reconstruction efficiency over largest-gap probing? The primary contrast is `cost - gap` in time-averaged radial RMSE over the first **50 modeled seconds**. Negative differences favor cost. This horizon was selected from the development pilot's shortest completed episode, 51.6427 seconds, rounded down before the held-out run. It includes shared initialization. Compute time is measured separately and is not part of modeled motion time.

The comparison also reports final radial RMSE at 128 touches, all four alternatives against gap, and separate noise-level results. Their intervals are marginal descriptive intervals, without a multiple-comparison correction; this is not a collection of independent significance tests.

- 20 new shape instances: four each of circle, ellipse, rectangle, Fourier outline, and narrow recess, using the existing declared generator families.
- Shape seeds `700000 + 10000*family_index + instance_index`; all generated parameters are stored explicitly. No shapes were selected or rejected based on reconstruction outcomes.
- Episode seeds `800000 + 10000*family_index + instance_index` determine candidate phase and policy randomness independently of the shape generator.
- Noise seeds 51001 and 51002, separately derived per episode, vary sensor noise without changing geometry, candidate phase, or random-policy order. Candidate/repeat noise is paired across policies. Noise levels share standardized draws.
- Correctly specified noise standard deviations 0, 0.005, and 0.015. No bias or contamination in this primary comparison.
- Five policies: random, sweep, gap, variance, cost. All use the same fixed GP, eight initial touches, 128 total touches, 256 candidates, 256 integration points, and 4096 truth-only evaluation points.
- GP mean 0.525, signal standard deviation 0.2, Matérn length scale 0.4, jitter initially 1e-10. No hyperparameter fitting, coverage overrides, or early stopping.
- 20 × 3 × 2 × 5 = **600 episodes**. At zero noise, the two noise repetitions coincide; averaging them within shapes does not increase the inferential sample size.

## Analysis fixed in advance

Integrate the stepwise RMSE curve exactly between completed observations. If an episode finishes before the time horizon, hold its final estimate. Plot time curves on a 201-point display grid; the scalar integral does not use that approximation.

Average the two noise repeats within each shape and noise level. For pooled results, also average the three noise levels within each shape. Compute each paired policy-minus-gap difference on that same shape. Average shapes within families, then give all five families equal weight.

For nominal 95% percentile bootstrap intervals, resample four shapes with replacement within each family, retaining all paired policy, noise-level, and repetition results belonging to each selected shape. Use 10,000 resamples and seed 91001. With only four shapes per family, intervals are approximate and cannot characterize unobserved shape diversity reliably. The population of interest is the specified family mixture, not arbitrary objects.

Report family/noise-level means as well as pooled results. Secondary diagnostics include sampled maximum error, pointwise interval coverage, interval width, motion time, distance, repeated-probe fraction, and per-episode 95th-percentile fitting/selection time. Record checkpoints at 8, 16, 32, 64, and 128 touches. Compute timing under concurrent execution is descriptive, not a hardware-independent ranking.

Retain every scheduled run. If a run fails or is missing, withhold paired inference instead of dropping the case silently. Preserve successful raw logs and failure details. Regenerating analysis reads saved results and never executes the simulator.

After the run, re-evaluate final predictions for all recess episodes on 8192 angles to check sensitivity to evaluation resolution. This diagnostic changes neither actions nor the frozen primary metrics. Show both a successful reconstruction and a failure when present. No claim of continuous maximum-error certification follows from either grid.

## Commands

```bash
uv run --frozen touch-explorer heldout --protocol configs/heldout.json --workers 2 --output results/heldout
uv run --frozen touch-explorer analyze results/heldout
```

Two worker processes each limit BLAS to one thread. Existing output directories are preserved. The primary study does not rerun learned models or confidence stopping; their development results remain separate, and held-out extension comparisons remain future work.
