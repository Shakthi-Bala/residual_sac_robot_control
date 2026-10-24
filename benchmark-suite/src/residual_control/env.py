import copy
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import gym
import mujoco
import numpy as np

from residual_control import ROOT
from residual_control.control import (
    HOME,
    CommandDelay,
    Limits,
    cartesian_controller,
    compose_command,
    filter_command,
)


@dataclass(frozen=True)
class Scenario:
    seed: int
    task: str = "tracking"
    payload: float = 0.0
    delay: int = 0

    def __post_init__(self):
        if self.task not in ("reach", "tracking"):
            raise ValueError(f"Unknown task: {self.task}")
        if not np.isfinite(self.payload) or self.payload < 0:
            raise ValueError("Payload must be finite and nonnegative")
        if not isinstance(self.delay, (int, np.integer)) or self.delay < 0:
            raise ValueError("Delay must be a nonnegative number of control steps")


class UR16eTrackingEnv(gym.Env):
    metadata = {"render_fps": 25}
    observation_shape = (87,)
    dt = 0.02
    horizon = 250
    history_length = 9

    def __init__(
        self,
        mode="residual",
        kp=4.0,
        ki=0.5,
        assumed_payload=1.0,
        randomize=True,
        presentation=False,
    ):
        if mode not in ("controller", "standalone", "residual"):
            raise ValueError(f"Unknown control mode: {mode}")
        self.mode = mode
        self.kp, self.ki = kp, ki
        self.randomize = randomize
        self.limits = Limits()
        path = str(ROOT / "models/ur16e.xml")
        if presentation:
            xml = ET.parse(path).getroot()
            xml.find("compiler").set("meshdir", str(ROOT / "models/assets"))
            ET.SubElement(
                xml.find("asset"),
                "texture",
                type="skybox",
                builtin="gradient",
                rgb1=".86 .89 .94",
                rgb2=".97 .98 1",
                width="512",
                height="512",
            )
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
        else:
            self.model = mujoco.MjModel.from_xml_path(path)
        self.nominal = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.nominal_data = mujoco.MjData(self.nominal)
        self.tcp_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.payload_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
        self.nominal.body_mass[self.payload_id] = 0.8 + assumed_payload
        self.nominal.body_inertia[self.payload_id] = 0.001 + assumed_payload * 0.00064
        mujoco.mj_setConst(self.nominal, self.nominal_data)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, self.observation_shape, dtype=np.float32
        )
        self.rng = np.random.default_rng()
        self.jacobian = np.zeros((3, 6))
        self.rotation_jacobian = np.zeros((3, 6))
        self.servo_gains = np.array([160.0, 160.0, 100.0, 25.0, 25.0, 20.0])
        self.physics_steps = round(self.dt / self.model.opt.timestep)
        self.renderer = None

    def target(self, t):
        if self.scenario.task == "reach" or t <= 1.0:
            return self.center.copy(), np.zeros(3)
        t -= 1.0
        ramp = 1.0 - np.exp(-t / 0.4)
        ramp_dot = np.exp(-t / 0.4) / 0.4
        phase = self.frequency * t
        position = self.center + self.amplitude * ramp * np.sin(phase)
        velocity = self.amplitude * (
            ramp_dot * np.sin(phase) + ramp * self.frequency * np.cos(phase)
        )
        return position, velocity

    @property
    def tcp(self):
        return self.data.site_xpos[self.tcp_id].copy()

    def reset(self, seed=None, scenario=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if scenario is None:
            scenario = Scenario(
                seed=int(self.rng.integers(0, 2**31)),
                task=str(self.rng.choice(["reach", "tracking"])),
                payload=float(self.rng.uniform(0.0, 3.0)) if self.randomize else 0.0,
                delay=int(self.rng.integers(0, 6)) if self.randomize else 0,
            )
        self.scenario = scenario
        rng = np.random.default_rng(scenario.seed)
        mujoco.mj_resetData(self.model, self.data)
        self.model.body_mass[self.payload_id] = 0.8 + scenario.payload
        self.model.body_inertia[self.payload_id] = 0.001 + scenario.payload * 0.00064
        mujoco.mj_setConst(self.model, self.data)
        self.data.qpos[:] = HOME + rng.uniform(-0.08, 0.08, 6)
        initial = self.data.qpos.copy()
        target_q = HOME + rng.uniform(-1.0, 1.0, 6) * np.array([0.25, 0.2, 0.25, 0.1, 0.1, 0.1])
        self.data.qpos[:] = target_q
        mujoco.mj_forward(self.model, self.data)
        self.center = self.tcp
        self.data.qpos[:] = initial
        self.data.qvel[:] = 0.0
        self.amplitude = (
            rng.uniform(0.7, 1.0, 3) * np.array([0.07, 0.06, 0.04]) * rng.choice([-1.0, 1.0], 3)
        )
        self.frequency = 2 * np.pi * rng.uniform(0.15, 0.25) * np.array([1.0, 0.8, 0.6])
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
        obs = np.concatenate(
            [
                np.sin(self.data.qpos),
                np.cos(self.data.qpos),
                self.data.qvel / self.limits.velocity,
                (target - self.tcp) / 0.2,
                self.tcp,
                target,
                velocity / 0.2,
                self.integral / 0.2,
                self.history.flatten() / self.limits.velocity,
            ]
        )
        assert obs.shape == self.observation_space.shape
        return obs.astype(np.float32)

    def step(self, action):
        target, target_velocity = self.target(self.steps * self.dt)
        error = target - self.tcp
        self.integral = np.clip(self.integral + error * self.dt, -0.15, 0.15)
        mujoco.mj_jacSite(self.model, self.data, self.jacobian, self.rotation_jacobian, self.tcp_id)
        base = cartesian_controller(
            self.jacobian, error, target_velocity, self.integral, self.data.qpos, self.kp, self.ki
        )
        requested, fallback = compose_command(self.mode, action, base, self.limits)
        issued = filter_command(requested, self.data.qpos, self.previous, self.dt, self.limits)
        acceleration = (issued - self.previous) / self.dt
        delayed = self.delay.push(issued)
        self.history = np.roll(self.history, 1, axis=0)
        self.history[0] = issued
        for _ in range(self.physics_steps):
            self.nominal_data.qpos[:] = self.data.qpos
            self.nominal_data.qvel[:] = self.data.qvel
            mujoco.mj_forward(self.nominal, self.nominal_data)
            applied = filter_command(delayed, self.data.qpos, delayed, self.dt, self.limits)
            torque = self.servo_gains * (applied - self.data.qvel) + self.nominal_data.qfrc_bias
            self.data.ctrl[:] = np.clip(
                torque, self.model.actuator_ctrlrange[:, 0], self.model.actuator_ctrlrange[:, 1]
            )
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        target, _ = self.target(self.steps * self.dt)
        distance = float(np.linalg.norm(target - self.tcp))
        failed = (
            not np.isfinite(self.data.qpos).all()
            or self.tcp[2] < 0.12
            or np.any(self.data.qpos < self.limits.lower - 0.03)
            or np.any(self.data.qpos > self.limits.upper + 0.03)
        )
        self.failures += int(failed)
        self.fallbacks += int(fallback)
        reward = (
            -10.0 * distance
            - 0.002 * np.square(issued).sum()
            - 0.0001 * np.square(acceleration).sum()
        )
        reward += 0.1 * np.exp(-np.square(distance / 0.02)) - 5.0 * failed
        self.trace.append(
            {
                "time": self.steps * self.dt,
                "error": distance,
                "acceleration_squared": float(np.square(acceleration).mean()),
                "command": issued.copy(),
                "tcp": self.tcp,
                "target": target.copy(),
            }
        )
        self.previous = issued
        done = failed or self.steps >= self.horizon
        return (
            self.observe(),
            float(reward),
            bool(done),
            {"failed": bool(failed), "TimeLimit.truncated": bool(done and not failed)},
        )

    def metrics(self):
        error = np.array([row["error"] for row in self.trace])
        times = np.array([row["time"] for row in self.trace])
        after = times >= 1.0
        window = error[after] if after.any() else error
        stable = np.maximum.accumulate(error[::-1])[::-1] < 0.02
        success = (
            np.all(error[-min(25, len(error)) :] < 0.02)
            if self.scenario.task == "reach"
            else np.mean(window < 0.02) >= 0.9
        )
        return {
            "success": float(success and not self.failures),
            "final_error_m": float(error[-1]),
            "tracking_rmse_m": float(np.sqrt(np.mean(window**2))),
            "settling_time_s": float(times[np.flatnonzero(stable)[0]])
            if stable.any()
            else self.horizon * self.dt,
            "settled": float(stable.any()),
            "command_acceleration_rms": float(
                np.sqrt(np.mean([r["acceleration_squared"] for r in self.trace]))
            ),
            "failure": float(bool(self.failures)),
            "fallbacks": self.fallbacks,
        }

    def render(self):
        from residual_control.rendering import ArmRenderer

        if self.renderer is None:
            self.renderer = ArmRenderer(self.model)
        self.model.site_pos[self.target_id] = self.target(self.steps * self.dt)[0]
        # Forward dynamics updates solver warm-start state; rendering must use a copy.
        render_data = copy.copy(self.data)
        mujoco.mj_forward(self.model, render_data)
        return self.renderer.render(render_data)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
