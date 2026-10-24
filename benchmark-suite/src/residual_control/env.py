from dataclasses import dataclass
from pathlib import Path

import gym
import mujoco
import numpy as np

from residual_control.control import HOME, CommandDelay, Limits, cartesian_controller, compose_command, filter_command


ROOT = Path(__file__).resolve().parents[3] / 'residual-control'
JOINTS = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')


@dataclass(frozen=True)
class Scenario:
    seed: int
    task: str = 'tracking'
    payload: float = 0.
    delay: int = 0


class UR16eTrackingEnv(gym.Env):
    metadata = {'render_fps': 25}
    dt = .02
    horizon = 250
    history_length = 9

    def __init__(self, mode='residual', kp=4., ki=.5, assumed_payload=1., randomize=True):
        self.mode = mode
        self.kp, self.ki = kp, ki
        self.randomize = randomize
        self.limits = Limits()
        path = str(ROOT / 'models/ur16e.xml')
        self.model = mujoco.MjModel.from_xml_path(path)
        self.nominal = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.nominal_data = mujoco.MjData(self.nominal)
        self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'tcp')
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'target')
        self.payload_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'payload')
        self.nominal.body_mass[self.payload_id] = .8 + assumed_payload
        self.nominal.body_inertia[self.payload_id] = .001 + assumed_payload * .00064
        mujoco.mj_setConst(self.nominal, self.nominal_data)
        self.action_space = gym.spaces.Box(-1., 1., (6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (87,), dtype=np.float32)
        self.rng = np.random.default_rng()
        self.jacobian = np.zeros((3, 6))
        self.rotation_jacobian = np.zeros((3, 6))
        self.renderer = None

    def target(self, t):
        if self.scenario.task == 'reach' or t <= 1.:
            return self.center.copy(), np.zeros(3)
        t -= 1.
        ramp = 1. - np.exp(-t / .4)
        ramp_dot = np.exp(-t / .4) / .4
        phase = self.frequency * t
        position = self.center + self.amplitude * ramp * np.sin(phase)
        velocity = self.amplitude * (ramp_dot * np.sin(phase) + ramp * self.frequency * np.cos(phase))
        return position, velocity

    @property
    def tcp(self):
        return self.data.site_xpos[self.tcp_id].copy()

    def reset(self, seed=None, scenario=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if scenario is None:
            scenario = Scenario(
                seed=int(self.rng.integers(0, 2 ** 31)),
                task=str(self.rng.choice(['reach', 'tracking'])),
                payload=float(self.rng.uniform(0., 3.)) if self.randomize else 0.,
                delay=int(self.rng.integers(0, 6)) if self.randomize else 0,
            )
        self.scenario = scenario
        rng = np.random.default_rng(scenario.seed)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_mass[self.payload_id] = .8 + scenario.payload
        self.model.body_inertia[self.payload_id] = .001 + scenario.payload * .00064
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:] = HOME + rng.uniform(-.08, .08, 6)
        initial = self.data.qpos.copy()
        target_q = HOME + rng.uniform(-1., 1., 6) * np.array([.25, .2, .25, .1, .1, .1])
        self.data.qpos[:] = target_q
        mujoco.mj_forward(self.model, self.data)
        self.center = self.tcp
        self.data.qpos[:] = initial
        self.data.qvel[:] = 0.
        self.amplitude = rng.uniform(.7, 1., 3) * np.array([.07, .06, .04]) * rng.choice([-1., 1.], 3)
        self.frequency = 2 * np.pi * rng.uniform(.15, .25) * np.array([1., .8, .6])
        self.delay = CommandDelay(scenario.delay)
        self.history = np.zeros((self.history_length, 6))
        self.integral = np.zeros(3)
        self.previous = np.zeros(6)
        self.steps = 0
        self.trace = []
        self.failures = 0
        self.fallbacks = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def observe(self):
        target, velocity = self.target(self.steps * self.dt)
        obs = np.concatenate([
            np.sin(self.data.qpos), np.cos(self.data.qpos), self.data.qvel / self.limits.velocity,
            (target - self.tcp) / .2, self.tcp, target, velocity / .2,
            self.integral / .2, self.history.flatten() / self.limits.velocity,
        ])
        assert obs.shape == self.observation_space.shape
        return obs.astype(np.float32)

    def step(self, action):
        target, target_velocity = self.target(self.steps * self.dt)
        error = target - self.tcp
        self.integral = np.clip(self.integral + error * self.dt, -.15, .15)
        mujoco.mj_jacSite(self.model, self.data, self.jacobian, self.rotation_jacobian, self.tcp_id)
        base = cartesian_controller(self.jacobian, error, target_velocity, self.integral, self.data.qpos, self.kp, self.ki)
        requested, fallback = compose_command(self.mode, action, base, self.limits)
        issued = filter_command(requested, self.data.qpos, self.previous, self.dt, self.limits)
        acceleration = (issued - self.previous) / self.dt
        delayed = self.delay.push(issued)
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = issued
        gains = np.array([160., 160., 100., 25., 25., 20.])
        for _ in range(round(self.dt / self.model.opt.timestep)):
            self.nominal_data.qpos[:] = self.data.qpos
            self.nominal_data.qvel[:] = self.data.qvel
            mujoco.mj_forward(self.nominal, self.nominal_data)
            applied = filter_command(delayed, self.data.qpos, delayed, self.dt, self.limits)
            torque = gains * (applied - self.data.qvel) + self.nominal_data.qfrc_bias
            self.data.ctrl[:] = np.clip(torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        target, _ = self.target(self.steps * self.dt)
        distance = float(np.linalg.norm(target - self.tcp))
        failed = (not np.isfinite(self.data.qpos).all() or self.tcp[2] < .12
                  or np.any(self.data.qpos < self.limits.lower - .03)
                  or np.any(self.data.qpos > self.limits.upper + .03))
        self.failures += int(failed)
        self.fallbacks += int(fallback)
        reward = -10. * distance - .002 * np.square(issued).sum() - .0001 * np.square(acceleration).sum()
        reward += .1 * np.exp(-np.square(distance / .02)) - 5. * failed
        self.trace.append({'time': self.steps * self.dt, 'error': distance, 'acceleration_squared': float(np.square(acceleration).mean()), 'command': issued.copy(), 'tcp': self.tcp, 'target': target.copy()})
        self.previous = issued
        done = failed or self.steps >= self.horizon
        return self.observe(), float(reward), bool(done), {'failed': bool(failed), 'TimeLimit.truncated': bool(done and not failed)}

    def metrics(self):
        error = np.array([row['error'] for row in self.trace])
        times = np.array([row['time'] for row in self.trace])
        after = times >= 1.
        window = error[after] if after.any() else error
        stable = np.maximum.accumulate(error[::-1])[::-1] < .02
        success = (np.all(error[-min(25, len(error)):] < .02) if self.scenario.task == 'reach'
                   else np.mean(window < .02) >= .9)
        return {
            'success': float(success and not self.failures),
            'final_error_m': float(error[-1]),
            'tracking_rmse_m': float(np.sqrt(np.mean(window ** 2))),
            'settling_time_s': float(times[np.flatnonzero(stable)[0]]) if stable.any() else self.horizon * self.dt,
            'settled': float(stable.any()),
            'command_acceleration_rms': float(np.sqrt(np.mean([r['acceleration_squared'] for r in self.trace]))),
            'failure': float(bool(self.failures)), 'fallbacks': self.fallbacks,
        }

    def render(self):
        from residual_control.rendering import ArmRenderer
        if self.renderer is None:
            self.renderer = ArmRenderer(self.model)
        self.model.site_pos[self.target_id] = self.target(self.steps * self.dt)[0]
        mujoco.mj_forward(self.model, self.data)
        return self.renderer.render(self.data)

    def close(self):
        if self.renderer:
            self.renderer.close()
