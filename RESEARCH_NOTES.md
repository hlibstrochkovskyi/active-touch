# Research notes

Reviewed on 2026-09-15. These notes distinguish findings in the sources from design decisions proposed for this project. This is a focused reading list, not an exhaustive literature review or a claim of research novelty.

## 1. What the original recommendation got wrong

The earlier conversation cited Yi et al. as the foundation for an implicit-surface implementation. Their 2016 paper actually chooses an **explicit surface**, predicting one Cartesian coordinate from the other two. In Section III-B they explain that this avoids artificial interior and exterior constraints. Their action rule selects high predicted standard deviation. The project will use a different explicit parameterization: radius as a function of angle. This is an adaptation of the modeling approach, not a reproduction of their experiment.

The same paper's toy experiment discusses early overconfidence caused by a length scale that is too large. That motivates testing premature confidence here, but does not establish how our implementation will perform. [Yi et al., *Active Tactile Object Exploration with Gaussian Processes*, IROS 2016, Sections III-B, III-C and IV-A](https://users.cs.utah.edu/~thermans/papers/yi-iros2016-gp-active-touch.pdf).

## 2. Travel cost is already an established research idea

Matsubara and Shibata combine shape uncertainty with travel cost and compare against passive exploration and uncertainty-only selection. Their work includes 2D/3D simulation and robot experiments. We will borrow the question of time efficiency, but use a known circular transit path and a simple greedy rule. Their reported improvements are not promised results for this project. The accessible abstract, introduction, and section summaries were reviewed; this plan does not claim a full reproduction of their method. [*Active tactile exploration with uncertainty and travel cost for fast shape estimation of unknown objects*, Robotics and Autonomous Systems, 2017](https://doi.org/10.1016/j.robot.2017.01.014).

Consequently, “GP uncertainty plus travel cost” is not our novelty claim. The personal project contribution is a transparent implementation and a controlled study of sampling, model mismatch, missed features, and stopping behavior.

## 3. Why implicit surfaces are a later step

Williams and Fitzgibbon formulate surfaces as level sets of a GP. Their examples use on-surface and signed off-surface constraints. A zero-mean GP trained only on zero-valued contact observations has an identically zero posterior mean; those data alone do not produce a useful signed inside/outside model. A binary contact switch also does not directly report a surface normal or a signed-distance value. These are measurement-model issues, not something solved by choosing a more powerful optimizer. [*Gaussian Process Implicit Surfaces*, 2007](https://gpss.cc/gpip/abstract/owilliams.pdf).

The optional GPIS extension therefore requires an explicit added sensor assumption or a different likelihood. The main project avoids manufactured interior measurements by directly modeling the measured radial coordinate.

## 4. GP mathematics and implementation

The GP posterior is the mathematical basis for interpolation and uncertainty. Use Cholesky solves rather than explicitly calculating a matrix inverse. Read the regression derivation and Algorithm 2.1; the complete textbook is not a prerequisite. [Rasmussen and Williams, *Gaussian Processes for Machine Learning*, Chapter 2](https://gaussianprocess.org/gpml/chapters/RW2.pdf).

Kernel choice controls how measurements influence nearby locations. The plan restricts a Matérn kernel on the plane to the unit circle, yielding a periodic angular model. [GPML, Chapter 4](https://gaussianprocess.org/gpml/chapters/RW4.pdf).

Marginal likelihood can select hyperparameters, but fitting them from a small adaptive dataset does not make uncertainty automatically reliable. The fixed-parameter baseline and the learned-parameter experiment must be reported separately. [GPML, Chapter 5](https://gaussianprocess.org/gpml/chapters/RW5.pdf).

For the implementation, `GaussianProcessRegressor` supplies posterior mean, standard deviation, and covariance. `alpha` is a diagonal variance contribution, and `optimizer=None` disables parameter optimization. The first implementation uses that library and tests only the small amount of custom mathematics. [Official regressor reference](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html), [official Matérn reference](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.Matern.html).

## 5. Information gain is not interchangeable with variance reduction

Krause, Singh, and Guestrin study sensor placement under a mutual-information objective and provide approximation results under their formulation. Our proposed acquisition is integrated posterior variance reduction divided by estimated motion time. It is a different objective with moving, state-dependent costs; those approximation guarantees do not carry over. [*Near-Optimal Sensor Placements in Gaussian Processes*, JMLR 2008](https://www.jmlr.org/papers/v9/krause08a.html).

## 6. Context beyond the core

A 2024 Bayesian tactile framework combines recognition, pose estimation, and shape transfer with particle filtering and GPIS. It is useful context for what a larger project could add. Those capabilities introduce extra latent variables and are deliberately outside this project's core deliverable. [*A Bayesian Framework for Active Tactile Object Recognition, Pose Estimation and Shape Transfer Learning*](https://arxiv.org/abs/2409.06912).

For a later polygon simulator, Shapely provides line/polygon intersection operations. It should remain confined to the world and evaluator; policy code must never query hidden geometry. [Official intersection documentation](https://shapely.readthedocs.io/en/stable/reference/shapely.intersection.html).

## Reading order

1. Yi et al., especially the explicit representation and acquisition rule.
2. GPML Chapter 2, alongside one tiny regression example.
3. The Matérn API and the relevant part of GPML Chapter 4.
4. Matsubara and Shibata's problem formulation and comparisons.
5. GPML Chapter 5 when adding hyperparameter fitting.
6. Williams and Fitzgibbon only before deciding whether to build GPIS.

Read to answer an implementation question, then build the corresponding component. Do not postpone the first simulation until every source has been read.
