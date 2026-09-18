# Sensor mismatch development study — 2026-09-18

All 48 episodes completed without execution failures or optimizer fallbacks. Constant sensor bias defeated the stopping audits in all six tested cases. This is a small development study, not a measured deployment failure rate.

## Reproduce

```bash
uv run --frozen touch-explorer mismatch --config configs/mismatch.toml --seeds 2 --output results/mismatch
```

Implementation revision: `f910138`. All episodes recorded that revision with a clean working tree. Raw outputs are in `results/mismatch-2026-09-18/`, excluded from Git. The manifest records conditions and library versions. Summed episode wall time was about 59 seconds on this machine, excluding the final aggregate plots.

Six cases: two generated circles, two ellipses, and two narrow recesses; one noise realization per instance. Root seed 3000. Each case runs four sensor conditions under two methods. Candidate/repeat-indexed random streams pair measurements across methods and preserve nominal noise draws across conditions. Adaptive action sequences can subsequently diverge.

Both methods learn GP parameters and use largest-gap exploration every fifth post-initialization action. One completes 128 touches; the other uses the existing guarded stopping rule. Settings match the previous reliability study: eight initial touches, 256 candidates, 128 integration points, 2048 evaluation points, and assumed noise standard deviation 0.005. No model or stopping thresholds were tuned after seeing these results.

## Sensor model

The simulated reading is

\[
y_i = r(\theta_i) + b + s\sigma z_i + B_i\tau w_i,
\]

where independent standard normal variables are \(z_i,w_i\), and \(B_i\) is Bernoulli with probability \(p\). Thus the error has mean \(b\) and variance \(s^2\sigma^2+p\tau^2\). The GP always assumes zero bias and variance \(\sigma^2\). Neither readings nor outliers are clipped.

| Condition | Bias b | Noise scale s | Contamination p | Outlier std τ |
|---|---:|---:|---:|---:|
| Nominal | 0 | 1 | 0 | 0 |
| Bias | 0.015 | 1 | 0 | 0 |
| Noise | 0 | 3 | 0 | 0 |
| Outliers | 0 | 1 | 0.05 | 0.05 |

Outlier contamination is a Gaussian mixture, with overall standard deviation approximately 0.01225; individual contamination draws need not be large. These conditions are illustrative stress levels, not a calibrated model of a particular sensor. Angles, physical contact, and motion remain exact. Fault parameters never enter policy decisions directly.

## Results

Quality requires both radial RMSE ≤ 0.01 and sampled maximum radial error ≤ 0.03. A false stop means the controller declared confidence while either limit failed. Budget exhaustion is reported separately from confidence stopping.

| Condition / method | Mean touches | Mean RMSE | Quality met | False / confidence stops | Budget exhausted |
|---|---:|---:|---:|---:|---:|
| Nominal / budget | 128 | 0.003005 | 6/6 | No stops | 6/6 |
| Nominal / guarded | 34 | 0.009720 | 4/6 | 2/6 | 0/6 |
| Bias / budget | 128 | 0.015385 | 0/6 | No stops | 6/6 |
| Bias / guarded | 34 | 0.019290 | 0/6 | 6/6 | 0/6 |
| Noise / budget | 128 | 0.014419 | 0/6 | No stops | 6/6 |
| Noise / guarded | 95.5 | 0.014227 | 0/6 | 3/3 | 3/6 |
| Outliers / budget | 128 | 0.009149 | 0/6 | No stops | 6/6 |
| Outliers / guarded | 34 | 0.014416 | 1/6 | 5/6 | 0/6 |

At the fixed 128-touch budget, the paired mean RMSE increase relative to nominal sensing was 0.012380 for bias, 0.011414 for higher noise, and 0.006144 for outliers. These are descriptive differences across six matched cases; no population confidence claim follows.

Nominal guarded failures were both recesses. Bias also broke the four previously successful circle/ellipse cases. Mean pointwise coverage under bias fell to 1.8% at full budget and 10.0% with guarded stopping, despite similar interval widths to nominal sensing. The model became confident about a shifted outline.

Higher noise exhausted three guarded budgets; all three confidence stops were false. Outlier runs illustrate why mean RMSE alone is insufficient: full-budget mean RMSE was below 0.01, but every case failed the combined accuracy criteria. Additional measurements can expose further contamination, so more touches need not improve every case.

## Inspect a bias failure

```bash
uv run --frozen touch-explorer replay results/mismatch-2026-09-18/circle-3000-bias-guarded
xdg-open results/mismatch-2026-09-18/circle-3000-bias-guarded/replay.html
```

This run stopped after 34 touches with RMSE 0.015737. Toggle truth to see the offset. A constant bias and a shifted radial surface can produce identical readings, so repeated touches and self-consistency audits cannot generally separate them without calibration information.

The implementation now measures these failure modes; it does not correct bias or use a robust likelihood. Keep these cases for development. The final evaluation still needs independently varied noise seeds, held-out shapes, and paired statistical analysis. Any calibration or robust-inference extension should be a separately tested variant.
