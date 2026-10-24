# How the residual-control experiment works

## From position error to joint velocity

The robot has six joint angles, but the task specifies a three-dimensional tool position. Forward kinematics maps joint angles to that position. The translational Jacobian maps small joint velocities to tool velocity:

\[
\dot x = J(q)\dot q.
\]

The conventional controller computes a desired tool velocity from position error, accumulated error, and reference velocity. It uses a damped inverse to turn that into joint velocity:

\[
v_{PI}=J^T(JJ^T+\lambda^2I)^{-1}(k_pe+k_i\int e\,dt+\dot x_{ref})
 +(I-J^\#J)k_{posture}(q_{home}-q).
\]

Damping reduces extreme commands near singular configurations. The posture term discourages drift along joint motions that have little effect on tool position. With damping, this projection is approximate. Integral action compensates persistent error; the implementation clips the integral to prevent unrestricted accumulation.

## What SAC learns

Standalone SAC maps the observation to a normalized six-dimensional action, scaled to a joint velocity command. Residual SAC scales its action to a smaller correction and adds it to PI:

\[
v_{standalone}=0.7a,\qquad
v_{residual}=v_{PI}+0.2a,\quad a\in[-1,1]^6.
\]

Both commands pass through the same filters and servo. The residual bound limits the correction before final filtering; it does not mean that joint tracking error, torque, or Cartesian error is bounded by that number.

The initial actor has zero mean, so deterministic residual evaluation initially reproduces PI. Training samples a stochastic policy, so its early actions still explore. Residual learning changes the search problem: it can adjust an existing solution, while standalone SAC must discover one. That advantage depends on the quality and cost of the existing controller.

## Soft Actor-Critic

SAC is off-policy: it reuses transitions from a replay buffer. Two critics learn discounted returns, and a stochastic actor learns actions with high estimated return while retaining entropy. The entropy-adjusted target is

\[
y=r+\gamma(1-d)\left[\min_i Q_{target,i}(s',a')-\alpha\log\pi(a'|s')\right].
\]

The actor minimizes

\[
\mathbb E[\alpha\log\pi(a|s)-\min_iQ_i(s,a)].
\]

The minimum reduces the optimistic estimates that can arise when a policy exploits critic errors. Target critics change slowly using Polyak averaging. This experiment uses discount 0.99, target update coefficient 0.005, Adam learning rate 0.0003, and two 128-unit hidden layers. The policy uses a Gaussian followed by tanh; its log probability includes the change of variables. Temperature is learned against a target entropy of -6 in normalized action coordinates.

Time limits bootstrap the critic because the underlying task need not end at the recording boundary. Joint-envelope/height failures are terminal and do not bootstrap. Evaluation uses tanh of the Gaussian mean, with no exploration sampling and no parameter updates.

## Delay, payload, and partial observations

A FIFO delays commanded velocities by whole 20 ms control ticks. Added payload changes actual MuJoCo mass and inertia. The servo compensates dynamics using a separate nominal model whose payload estimate stays fixed. A torque-limited proportional velocity servo then drives the physical model.

The policy sees joint state, target information, integrated error, and recent issued commands. It does not receive payload mass or delay labels. Recent commands can help infer delayed effects, but finite history is not proof that the observation is Markov. A recurrent policy or explicit system identification would be a separate extension.

Domain randomization exposes both learned methods to payloads and delays during training. Holding out target seeds tests target generalization. Testing 5 kg payload and 160 ms delay probes extrapolation beyond the training ranges. Neither test establishes unrestricted robustness.

## Reward versus evaluation

The reward penalizes position error, issued-command magnitude, and changes in issued commands. It adds a small proximity bonus and a terminal failure penalty. These terms guide learning; success is evaluated separately using a 2 cm tolerance.

RMSE measures sustained tracking error; final error measures only the endpoint. Settling time asks when the error enters tolerance and stays there. Command acceleration RMS measures command smoothness, not mechanical joint acceleration. Failure rate must be reported alongside error because a short failed episode can otherwise appear deceptively good on a time average.

## What a fair conclusion requires

All methods share target trajectories, perturbations, filters, and low-level dynamics. Calibration chooses PI gains on separate targets. Validation selects learned checkpoints. Test targets are reserved for final measurement. Three training seeds measure some optimization variability; twelve test targets per condition measure target variability. These are different sources of uncertainty and are not interchangeable.

A residual policy that qualifies at step zero has inherited competence. It has not learned an improvement. This study therefore also requires 10% lower validation RMSE than PI without lower success, confirmed at two consecutive checkpoints. Controller calibration and validation interactions are reported separately from replay-generating SAC interactions. The 144,000-interaction calibration cost matters when comparing total data use.

A negative result is useful when it identifies where learning failed to add value. Retained initial checkpoints, poor standalone performance, late regressions, and out-of-distribution failures belong in the results rather than being removed by choosing favorable demonstrations.
