import numpy as np
import torch

from agents.sac_agent import SoftActorCritic
from infrastructure import pytorch_util as ptu
from networks.critics import StateActionCritic
from networks.policies import MLPPolicy


class TrackingSAC(SoftActorCritic):
    def actor_loss_reparametrize(self, obs):
        distribution = self.actor(obs)
        action = distribution.rsample()
        log_prob = distribution.log_prob(action)
        value = self.critic(obs, action).min(dim=0).values
        return -value.mean(), -log_prob.mean(), log_prob

    @torch.no_grad()
    def deterministic_action(self, observation):
        distribution = self.actor(ptu.from_numpy(observation)[None])
        mean = distribution.base_dist.base_dist.loc
        return torch.tanh(mean)[0].cpu().numpy()


def make_agent(observation_shape, config):
    def actor(shape, action_dim):
        policy = MLPPolicy(action_dim, shape[0], False, config['num_layers'], config['hidden_size'],
                           use_tanh=True, state_dependent_std=True)
        output = [layer for layer in policy.net if isinstance(layer, torch.nn.Linear)][-1]
        with torch.no_grad():
            output.weight.zero_()
            output.bias[:action_dim].zero_()
            output.bias[action_dim:].fill_(float(np.log(np.expm1(.24))))
        return policy

    def critic(shape, action_dim):
        return StateActionCritic(shape[0], action_dim, config['num_layers'], config['hidden_size'])

    def optimizer(parameters):
        return torch.optim.Adam(parameters, lr=config['learning_rate'])

    def schedule(optimizer):
        return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.)

    return TrackingSAC(
        observation_shape, 6, actor, optimizer, schedule, critic, optimizer, schedule,
        discount=.99, soft_target_update_rate=.005, num_critic_networks=2,
        target_critic_backup_type='min', use_entropy_bonus=True,
        temperature=config['temperature'], auto_tune_temperature=config['auto_tune_temperature'],
    )
