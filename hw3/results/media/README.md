## Policy demonstrations

Every trained configuration is shown below using its best checkpoint and visualization seed 10001. These individual rollouts illustrate behavior; evaluation averages are reported separately. Each GIF has a JSON file recording its checkpoint, seed, length, and return.

### Discrete action control

<table>
<tr>
<td width="33%"><strong>DQN · CartPole</strong><br><img src="cartpole.gif" width="250" alt="DQN · CartPole policy rollout"></td>
<td width="33%"><strong>Double DQN · LunarLander</strong><br><img src="lunarlander.gif" width="250" alt="Double DQN · LunarLander policy rollout"></td>
<td width="33%"><strong>Double DQN · MsPacman</strong><br><img src="mspacman.gif" width="250" alt="Double DQN · MsPacman policy rollout"></td>
</tr>
</table>

### HalfCheetah: fixed versus learned temperature

<table>
<tr>
<td width="50%"><strong>SAC · Fixed temperature</strong><br><img src="halfcheetah.gif" width="320" alt="SAC · Fixed temperature policy rollout"></td>
<td width="50%"><strong>SAC · Learned temperature</strong><br><img src="halfcheetah_autotune.gif" width="320" alt="SAC · Learned temperature policy rollout"></td>
</tr>
</table>

### Hopper: single versus clipped double Q

<table>
<tr>
<td width="50%"><strong>Hopper · Single Q</strong><br><img src="hopper_singleq.gif" width="320" alt="Hopper · Single Q policy rollout"></td>
<td width="50%"><strong>Hopper · Clipped double Q</strong><br><img src="hopper.gif" width="320" alt="Hopper · Clipped double Q policy rollout"></td>
</tr>
</table>

<details>
<summary>Sanity checks and target-update sweep — 5 additional GIFs</summary>

<table>
<tr>
<td width="50%"><strong>Pendulum · Fixed temperature</strong><br><img src="invertedpendulum_fixed.gif" width="320" alt="Pendulum · Fixed temperature policy rollout"></td>
<td width="50%"><strong>Pendulum · Learned temperature</strong><br><img src="invertedpendulum.gif" width="320" alt="Pendulum · Learned temperature policy rollout"></td>
</tr>
</table>

<table>
<tr>
<td width="33%"><strong>LunarLander · Target period 250</strong><br><img src="lunarlander_target250.gif" width="250" alt="LunarLander · Target period 250 policy rollout"></td>
<td width="33%"><strong>LunarLander · Target period 500</strong><br><img src="lunarlander_target500.gif" width="250" alt="LunarLander · Target period 500 policy rollout"></td>
<td width="33%"><strong>LunarLander · Target period 2000</strong><br><img src="lunarlander_target2000.gif" width="250" alt="LunarLander · Target period 2000 policy rollout"></td>
</tr>
</table>

</details>
