import unittest

import numpy as np
import torch
import yaml
from infrastructure import pytorch_util as ptu
from residual_control.agent import make_agent
from residual_control.control import (
    HOME,
    CommandDelay,
    Limits,
    compose_command,
    filter_command,
)
from residual_control.env import ROOT, Scenario, UR16eTrackingEnv
from residual_control.protocol import calibration_cases, test_cases, validation_cases


class ControlTest(unittest.TestCase):
    def test_residual_bound_and_fallback(self):
        limits = Limits()
        base = np.linspace(-0.2, 0.2, 6)
        command, fallback = compose_command("residual", np.full(6, 20.0), base, limits)
        np.testing.assert_allclose(command - base, limits.residual)
        self.assertFalse(fallback)
        command, fallback = compose_command("residual", np.full(6, np.nan), base, limits)
        np.testing.assert_array_equal(command, base)
        self.assertTrue(fallback)

    def test_delay_exactly_n_steps(self):
        queue = CommandDelay(3)
        for value in (1.0, 2.0, 3.0):
            np.testing.assert_array_equal(queue.push(np.full(6, value)), np.zeros(6))
        np.testing.assert_array_equal(queue.push(np.full(6, 4.0)), np.ones(6))
        np.testing.assert_array_equal(CommandDelay(0).push(np.ones(6)), np.ones(6))

    def test_velocity_acceleration_and_joint_envelope(self):
        limits = Limits()
        command = filter_command(np.full(6, 100.0), HOME, np.zeros(6), 0.02, limits)
        self.assertLessEqual(abs(command).max(), limits.acceleration * 0.02)
        command = filter_command(np.ones(6), limits.upper, np.ones(6), 0.02, limits)
        np.testing.assert_array_equal(command, np.zeros(6))

    def test_zero_residual_reproduces_baseline(self):
        baseline = UR16eTrackingEnv("controller")
        residual = UR16eTrackingEnv("residual")
        scenario = Scenario(101, "tracking", 3.0, 4)
        np.testing.assert_array_equal(
            baseline.reset(scenario=scenario), residual.reset(scenario=scenario)
        )
        for _ in range(20):
            a = baseline.step(np.zeros(6))
            b = residual.step(np.zeros(6))
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1], b[1])
        baseline.close()
        residual.close()

    def test_payload_changes_dynamics(self):
        a = UR16eTrackingEnv("standalone")
        b = UR16eTrackingEnv("standalone")
        a.reset(scenario=Scenario(7, payload=0.0))
        b.reset(scenario=Scenario(7, payload=5.0))
        a.step(np.zeros(6))
        b.step(np.zeros(6))
        self.assertGreater(np.linalg.norm(a.data.qvel - b.data.qvel), 0.01)
        a.close()
        b.close()

    def test_perturbation_labels_are_hidden(self):
        env = UR16eTrackingEnv("residual")
        nominal = env.reset(scenario=Scenario(42, "tracking", 0.0, 0))
        perturbed = env.reset(scenario=Scenario(42, "tracking", 5.0, 8))
        np.testing.assert_array_equal(nominal, perturbed)
        env.close()

    def test_time_limit_is_truncation(self):
        env = UR16eTrackingEnv("controller")
        env.reset(scenario=Scenario(42))
        env.horizon = 1
        _, _, done, info = env.step(np.zeros(6))
        self.assertTrue(done)
        self.assertTrue(info["TimeLimit.truncated"])
        self.assertFalse(info["failed"])
        env.close()

    def test_deterministic_action_matches_policy_median(self):
        torch.set_num_threads(1)
        ptu.init_gpu(use_gpu=False)
        config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
        agent = make_agent((87,), config)
        output = [layer for layer in agent.actor.net if isinstance(layer, torch.nn.Linear)][-1]
        with torch.no_grad():
            output.bias[:6].copy_(torch.linspace(-2.0, 2.0, 6))
            observation = np.random.default_rng(7).normal(size=87).astype(np.float32)
            distribution = agent.actor(ptu.from_numpy(observation)[None])
            median = distribution.base_dist.icdf(torch.full((1, 6), 0.5))
        np.testing.assert_array_equal(agent.deterministic_action(observation), median[0].numpy())

    def test_target_splits_and_initial_actor(self):
        groups = [
            {s.seed for s in calibration_cases()},
            {s.seed for s in validation_cases()},
            {s.seed for _, s in test_cases()},
        ]
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        torch.set_num_threads(1)
        ptu.init_gpu(use_gpu=False)
        config = yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())
        agent = make_agent((87,), config)
        np.testing.assert_array_equal(
            agent.deterministic_action(np.zeros(87, np.float32)), np.zeros(6)
        )


if __name__ == "__main__":
    unittest.main()
