from collections import deque
from dataclasses import dataclass

import numpy as np


HOME = np.array([0., -1.4, 1.4, -1.57, -1.57, 0.])


@dataclass(frozen=True)
class Limits:
    velocity: float = .7
    acceleration: float = 3.
    residual: float = .2
    margin: float = .08

    @property
    def lower(self):
        return HOME - np.array([.9, .65, .75, .9, .75, 1.])

    @property
    def upper(self):
        return HOME + np.array([.9, .65, .75, .9, .75, 1.])


def cartesian_controller(jacobian, error, target_velocity, integral, q, kp=4., ki=.5, damping=.04):
    inverse = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + damping ** 2 * np.eye(3), np.eye(3))
    command = inverse @ (kp * error + ki * integral + target_velocity)
    command += (np.eye(6) - inverse @ jacobian) @ (.3 * (HOME - q))
    return command


def compose_command(mode, action, baseline, limits):
    action = np.asarray(action, dtype=float)
    fallback = action.shape != (6,) or not np.isfinite(action).all()
    if mode == 'controller' or fallback:
        return np.asarray(baseline).copy(), fallback
    correction = np.clip(action, -1., 1.)
    if mode == 'residual':
        return baseline + limits.residual * correction, False
    if mode == 'standalone':
        return limits.velocity * correction, False
    raise ValueError(mode)


def filter_command(command, q, previous, dt, limits):
    command = np.nan_to_num(np.asarray(command), nan=0., posinf=0., neginf=0.)
    command = np.clip(command, -limits.velocity, limits.velocity)
    command = np.clip(command, previous - limits.acceleration * dt, previous + limits.acceleration * dt)
    positive = np.clip((limits.upper - q) / limits.margin, 0., 1.)
    negative = np.clip((q - limits.lower) / limits.margin, 0., 1.)
    return np.clip(command, -limits.velocity * negative, limits.velocity * positive)


class CommandDelay:
    def __init__(self, steps):
        if steps < 0:
            raise ValueError('Delay must be nonnegative')
        self.queue = deque([np.zeros(6) for _ in range(steps)])

    def push(self, command):
        self.queue.append(np.asarray(command).copy())
        return self.queue.popleft()
