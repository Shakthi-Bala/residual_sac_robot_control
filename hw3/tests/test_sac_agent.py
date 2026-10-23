import unittest

import torch

from agents.sac_agent import SoftActorCritic
from configs.sac_config import sac_config
from infrastructure import pytorch_util as ptu


class SACTest(unittest.TestCase):
    def setUp(self):
        ptu.init_gpu(use_gpu=False)
        torch.manual_seed(3)

    def make_agent(self, **kwargs):
        config = sac_config(
            'InvertedPendulum-v4', hidden_size=16, num_layers=1,
            use_soft_target_update=True, soft_target_update_rate=0.1,
            num_critic_networks=2, target_critic_backup_type='min', **kwargs,
        )
        return SoftActorCritic((4,), 1, **config['agent_kwargs'])

    def test_clipped_backup(self):
        agent = self.make_agent()
        values = torch.tensor([[3., 8.], [5., 2.]])
        torch.testing.assert_close(agent.q_backup_strategy(values), torch.tensor([[3., 2.], [3., 2.]]))

    def test_terminal_mask_and_target_gradients(self):
        agent = self.make_agent(use_entropy_bonus=False)
        with torch.no_grad():
            for critic, value in zip(agent.target_critics, (3., 4.)):
                for parameter in critic.parameters():
                    parameter.zero_()
                list(critic.parameters())[-1].fill_(value)
        obs = torch.zeros(2, 4)
        stats = agent.update_critic(obs, torch.zeros(2, 1), torch.tensor([1., 2.]), obs, torch.tensor([False, True]))
        self.assertAlmostEqual(stats['target_values'], (1 + .99 * 3 + 2) / 2, places=5)
        self.assertTrue(all(p.grad is None for p in agent.target_critics.parameters()))

    def test_actor_reparameterization_and_critic_isolation(self):
        agent = self.make_agent(use_entropy_bonus=False)
        old_actor = [p.detach().clone() for p in agent.actor.parameters()]
        old_critics = [p.detach().clone() for p in agent.critics.parameters()]
        info = agent.update_actor(torch.randn(32, 4))
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(old_actor, agent.actor.parameters())))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(old_critics, agent.critics.parameters())))
        self.assertTrue(all(p.grad is None for p in agent.critics.parameters()))
        self.assertFalse(info['log_prob'].requires_grad)

    def test_temperature_direction_and_detachment(self):
        for log_prob, expected_direction in [(4., 1), (-4., -1)]:
            agent = self.make_agent(auto_tune_temperature=True, temperature=.1)
            samples = torch.full((8,), log_prob, requires_grad=True)
            before = agent.get_temperature()
            agent.update_alpha(samples)
            self.assertGreater(expected_direction * (agent.get_temperature() - before), 0)
            self.assertIsNone(samples.grad)

    def test_polyak_update(self):
        agent = self.make_agent()
        with torch.no_grad():
            for p in agent.critics.parameters():
                p.fill_(2.)
            for p in agent.target_critics.parameters():
                p.zero_()
        agent.soft_update_target_critic(.1)
        for p in agent.target_critics.parameters():
            torch.testing.assert_close(p, torch.full_like(p, .2))

    def test_squashed_entropy_and_gradient(self):
        agent = self.make_agent()
        distribution = agent.actor(torch.zeros(4096, 4))
        entropy = agent.entropy(distribution)
        self.assertEqual(entropy.shape, (4096,))
        self.assertTrue(torch.isfinite(entropy).all())
        self.assertLess(entropy.mean().item(), .71)
        (-entropy.mean()).backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in agent.actor.parameters()))


if __name__ == '__main__':
    unittest.main()
