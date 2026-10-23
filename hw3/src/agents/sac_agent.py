from typing import Callable, Optional, Sequence, Tuple

import torch
from torch import nn
import numpy as np

from infrastructure import pytorch_util as ptu


class SoftActorCritic(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_dim: int,
        make_actor: Callable[[Tuple[int, ...], int], nn.Module],
        make_actor_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_actor_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_critic_optimizer: Callable[
            [torch.nn.ParameterList], torch.optim.Optimizer
        ],
        make_critic_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: Optional[int] = None,
        soft_target_update_rate: Optional[float] = None,
        # Actor-critic configuration
        num_critic_updates: int = 1,
        # Settings for multiple critics
        num_critic_networks: int = 1,
        target_critic_backup_type: str = "mean",  # One of "min" or "mean"
        # Soft actor-critic
        use_entropy_bonus: bool = False,
        temperature: float = 0.0,
        backup_entropy: bool = True,
        # Automatic temperature tuning (Section 3.5)
        auto_tune_temperature: bool = False,
        alpha_learning_rate: float = 3e-4,
    ):
        super().__init__()

        assert target_critic_backup_type in [
            "min",
            "mean",
        ], f"{target_critic_backup_type} is not a valid target critic backup type"

        assert (
            target_update_period is not None or soft_target_update_rate is not None
        ), "Must specify either target_update_period or soft_target_update_rate"

        self.actor = make_actor(observation_shape, action_dim)
        self.actor_optimizer = make_actor_optimizer(self.actor.parameters())
        self.actor_lr_scheduler = make_actor_schedule(self.actor_optimizer)

        self.critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.critic_optimizer = make_critic_optimizer(self.critics.parameters())
        self.critic_lr_scheduler = make_critic_schedule(self.critic_optimizer)
        self.target_critics = nn.ModuleList(
            [
                make_critic(observation_shape, action_dim)
                for _ in range(num_critic_networks)
            ]
        )

        self.observation_shape = observation_shape
        self.action_dim = action_dim
        self.discount = discount
        self.target_update_period = target_update_period
        self.target_critic_backup_type = target_critic_backup_type
        self.num_critic_networks = num_critic_networks
        self.use_entropy_bonus = use_entropy_bonus
        self.temperature = temperature
        self.num_critic_updates = num_critic_updates
        self.soft_target_update_rate = soft_target_update_rate
        self.backup_entropy = backup_entropy

        self.auto_tune_temperature = auto_tune_temperature
        if auto_tune_temperature:
            if temperature <= 0:
                raise ValueError("Automatic tuning requires a positive initial temperature")
            self.log_alpha = nn.Parameter(
                torch.tensor(np.log(temperature), dtype=torch.float32, device=ptu.device)
            )
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_learning_rate)
            self.target_entropy = -float(action_dim)

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_temperature(self) -> float:
        if self.auto_tune_temperature:
            return self.log_alpha.detach().exp().item()
        return self.temperature

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute the action for a given observation.
        """
        with torch.no_grad():
            observation = ptu.from_numpy(observation)[None]

            action_distribution: torch.distributions.Distribution = self.actor(observation)
            action: torch.Tensor = action_distribution.sample()

            assert action.shape == (1, self.action_dim), action.shape
            return ptu.to_numpy(action).squeeze(0)

    def critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) Q-values for the given state-action pair.
        """
        return torch.stack([critic(obs, action) for critic in self.critics], dim=0)

    def target_critic(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the (ensembled) target Q-values for the given state-action pair.
        """
        return torch.stack(
            [critic(obs, action) for critic in self.target_critics], dim=0
        )

    def q_backup_strategy(self, next_qs: torch.Tensor) -> torch.Tensor:
        """
        Handle Q-values from multiple different target critic networks to produce target values.

        For example:
         - for "mean", we average the Q-values across critics (single-Q baseline).
         - for "min", we take the minimum of the two critics' predictions (clipped double-Q).

        Parameters:
            next_qs (torch.Tensor): Q-values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FROM the different critics.
        Returns:
            torch.Tensor: Target values of shape (num_critics, batch_size).
                Leading dimension corresponds to target values FOR the different critics.
        """

        assert (
            next_qs.ndim == 2
        ), f"next_qs should have shape (num_critics, batch_size) but got {next_qs.shape}"
        num_critic_networks, batch_size = next_qs.shape
        assert num_critic_networks == self.num_critic_networks

        if self.target_critic_backup_type == "mean":
            next_qs = next_qs.mean(dim=0)
        elif self.target_critic_backup_type == "min":
            next_qs = next_qs.min(dim=0).values
        else:
            raise ValueError(
                f"Invalid critic backup strategy {self.target_critic_backup_type}"
            )

        # If our backup strategy removed a dimension, add it back in explicitly
        # (assume the target for each critic will be the same)
        if next_qs.shape == (batch_size,):
            next_qs = next_qs[None].expand((self.num_critic_networks, batch_size)).contiguous()

        assert next_qs.shape == (
            self.num_critic_networks,
            batch_size,
        ), next_qs.shape
        return next_qs

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ):
        """
        Update the critic networks by computing target values and minimizing Bellman error.
        """
        (batch_size,) = reward.shape

        with torch.no_grad():
            next_distribution = self.actor(next_obs)
            next_action = next_distribution.sample()
            next_qs = self.q_backup_strategy(self.target_critic(next_obs, next_action))
            if self.use_entropy_bonus and self.backup_entropy:
                next_qs -= self.get_temperature() * next_distribution.log_prob(next_action)
            target_values = reward + self.discount * (1.0 - done.float()) * next_qs

        q_values = self.critic(obs, action)
        assert q_values.shape == target_values.shape == (self.num_critic_networks, batch_size)
        loss = self.critic_loss(q_values, target_values)

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
        }

    def entropy(self, action_distribution: torch.distributions.Distribution):
        action = action_distribution.rsample()
        return -action_distribution.log_prob(action)

    def actor_loss_reparametrize(self, obs: torch.Tensor):
        distribution = self.actor(obs)
        action = distribution.rsample()
        log_prob = distribution.log_prob(action)
        q_values = self.critic(obs, action).mean(dim=0)
        return -q_values.mean(), -log_prob.mean(), log_prob

    def update_actor(self, obs: torch.Tensor):
        self.critics.requires_grad_(False)
        try:
            loss, entropy, log_prob = self.actor_loss_reparametrize(obs)
            if self.use_entropy_bonus:
                loss = loss - self.get_temperature() * entropy
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
        finally:
            self.critics.requires_grad_(True)
        return {
            "actor_loss": loss.item(),
            "entropy": entropy.item(),
            "log_prob": log_prob.detach(),
        }

    def update_alpha(self, log_prob: torch.Tensor):
        if not self.auto_tune_temperature:
            return {}
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        return {"alpha": self.get_temperature(), "alpha_loss": alpha_loss.item()}

    def update_target_critic(self):
        self.soft_update_target_critic(1.0)

    @torch.no_grad()
    def soft_update_target_critic(self, tau):
        for target, online in zip(self.target_critics.parameters(), self.critics.parameters()):
            target.lerp_(online, tau)

    def update(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ):
        """
        Update the actor and critic networks.
        """

        critic_infos = []
        for _ in range(self.num_critic_updates):
            info = self.update_critic(observations, actions, rewards, next_observations, dones)
            critic_infos.append(info)

        actor_info = self.update_actor(observations)

        # Update alpha (temperature) using dual gradient descent (Section 3.5)
        if self.auto_tune_temperature:
            alpha_info = self.update_alpha(actor_info["log_prob"])
        else:
            alpha_info = {}

        if self.soft_target_update_rate is not None:
            self.soft_update_target_critic(self.soft_target_update_rate)
        elif step % self.target_update_period == 0:
            self.update_target_critic()

        # Average the critic info over all of the steps
        critic_info = {
            k: np.mean([info[k] for info in critic_infos]) for k in critic_infos[0]
        }

        # Deal with LR scheduling
        self.actor_lr_scheduler.step()
        self.critic_lr_scheduler.step()

        # Remove log_prob from actor_info before returning (only needed for alpha update)
        if "log_prob" in actor_info:
            del actor_info["log_prob"]

        result = {
            **actor_info,
            **critic_info,
            **alpha_info,
            "actor_lr": self.actor_lr_scheduler.get_last_lr()[0],
            "critic_lr": self.critic_lr_scheduler.get_last_lr()[0],
        }

        # Always log temperature (whether fixed or learned)
        result["temperature"] = self.get_temperature()

        return result
