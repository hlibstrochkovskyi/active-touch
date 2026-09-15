# Touch Explorer

A planned laptop robotics project: a simulated probe reconstructs a fixed object's 2D outline from contact measurements, chooses its next approach, and tests whether its uncertainty is trustworthy.

**Status:** research and implementation plan only. No simulator or benchmark has been implemented or timed yet.

Read [the complete project plan](PROJECT_PLAN.md) for the sensing contract, mathematics, algorithms, software design, experiments, implementation sequence, and completion criteria. Read [the research notes](RESEARCH_NOTES.md) for source-specific findings and corrections to the initial idea.

## Recommended scope

The first complete version represents the outline as a radius versus angle. It supports convex objects and concave objects that are star-shaped about a known reference point. A probe approaches along a radius from a surrounding free circle, stops at first contact, and retracts. This makes the measurement and motion models precise while keeping the computation small.

The central question is:

> How much reconstruction accuracy per unit of motion time does uncertainty-guided probing buy, and when does the shape model become confident too early?

Compare random probing, a uniform sweep, largest-gap sampling, maximum GP variance, and predicted variance reduction per estimated action time. Test a periodic coverage safeguard on difficult examples. General implicit surfaces are a gated extension, with additional sensing requirements explained in the plan.

## Expected deliverables

- A reproducible simulator and a replayable visual demonstration.
- Tested GP estimation and action-selection mathematics.
- A benchmark with held-out shapes, explicit noise, strong baselines, and uncertainty diagnostics.
- A short report that includes both successful reconstructions and failures.

The working estimate is 65–95 focused hours for the complete core, including learning and debugging. The first visual milestone should be much smaller. These are planning estimates, not measured implementation results.
