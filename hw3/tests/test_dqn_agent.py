"""Numerical checks for the Section 2.4 Bellman update.

Run with: uv run python -m unittest discover -s tests -v
"""
import unittest

import numpy as np
import torch

from agents.dqn_agent import DQNAgent
from infrastructure import pytorch_util as ptu


class BasicDQNTest(unittest.TestCase):
    def setUp(self):
        ptu.init_gpu(use_gpu=False)
        self.agent = DQNAgent(
            observation_shape=(2,),
            num_actions=2,
            make_critic=lambda shape, n: torch.nn.Linear(2, n),
            make_optimizer=lambda params: torch.optim.SGD(params, lr=0.1),
            make_lr_schedule=lambda optimizer: torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0
            ),
            discount=0.9,
            target_update_period=2,
        )
        with torch.no_grad():
            self.agent.critic.weight.zero_()
            self.agent.critic.bias.copy_(torch.tensor([1.0, 2.0]))
            self.agent.target_critic.weight.zero_()
            self.agent.target_critic.bias.copy_(torch.tensor([3.0, 4.0]))

    def update(self, step):
        return self.agent.update(
            obs=torch.zeros(2, 2),
            action=torch.tensor([0, 1]),
            reward=torch.tensor([1.0, 2.0]),
            next_obs=torch.zeros(2, 2),
            done=torch.tensor([False, True]),
            step=step,
        )

    def test_epsilon_greedy(self):
        observation = np.zeros(2, dtype=np.float32)
        self.assertEqual(self.agent.get_action(observation, epsilon=0.0), 1)
        np.random.seed(7)
        actions = {self.agent.get_action(observation, epsilon=1.0) for _ in range(100)}
        self.assertEqual(actions, {0, 1})

    def test_bellman_target_and_selected_action_gradient(self):
        stats = self.update(step=1)
        # Targets: [1 + .9 * 4, 2]. Predictions: [1, 2].
        self.assertAlmostEqual(stats['target_values'], 3.3, places=5)
        self.assertAlmostEqual(stats['critic_loss'], 6.48, places=5)
        torch.testing.assert_close(self.agent.critic.bias, torch.tensor([1.36, 2.0]))
        self.assertTrue(all(p.grad is None for p in self.agent.target_critic.parameters()))

    def test_double_q_selects_online_and_evaluates_target(self):
        self.agent.use_double_q = True
        with torch.no_grad():
            self.agent.critic.bias.copy_(torch.tensor([5., 2.]))
        stats = self.update(step=1)
        self.assertAlmostEqual(stats['target_values'], (1 + .9 * 3 + 2) / 2, places=5)

    def test_target_copy_only_at_scheduled_step(self):
        self.update(step=1)
        torch.testing.assert_close(self.agent.target_critic.bias, torch.tensor([3.0, 4.0]))
        self.update(step=2)
        for online, target in zip(
            self.agent.critic.parameters(), self.agent.target_critic.parameters()
        ):
            torch.testing.assert_close(online, target)


if __name__ == '__main__':
    unittest.main()
