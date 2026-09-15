# First development pilot

50 episodes completed; no failed runs. This is a smoke/pilot comparison, not the final held-out study.

- 32 contacts per episode, including 8 shared initial contacts.
- 128 candidate angles, 128 integration angles, 1024 evaluation angles.
- Sensor noise standard deviation: 0.005 normalized length units.
- Two generated instances per family: circle, ellipse, rectangle, Fourier outline, narrow recess.
- Fixed Matérn 3/2 GP, length scale 0.4, signal deviation 0.2.
- Root seed 100; case seeds and full parameters are saved in each run.

| Policy | Mean final radial RMSE | Mean error over common motion-time horizon |
|---|---:|---:|
| gap | 0.007782 | 0.033021 |
| variance | 0.007782 | 0.034975 |
| sweep | 0.007782 | 0.036300 |
| cost | 0.007926 | 0.033365 |
| random | 0.012098 | 0.036775 |

The shared horizon is 51.6427 modeled seconds, selected descriptively as the shortest completed run. Error is held constant between observations. All ten cases have complete policy matches.

Gap, variance, and sweep reach the same final sampling grid here; paired noise makes their final reconstructions agree. Their sampling order changes the error accumulated during motion. The cost policy did not outperform gap in this pilot.

The execution environment measured roughly 0.10–0.18 seconds per episode, including saving raw data, with one BLAS thread. These are small 32-touch runs, not timings for the full 128-touch study. The p95 fit-plus-selection time for the cost policy was about 4.2 ms per action.

Reproduce with `configs/pilot.toml` and `--seeds 2`. Raw development output is in `results/first-pilot/` and is ignored by Git. No uncertainty or stopping guarantees follow from these results.
