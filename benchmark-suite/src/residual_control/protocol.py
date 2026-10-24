from residual_control.env import Scenario


CONDITIONS = {
    'nominal': (0., 0),
    'delay': (0., 5),
    'payload': (3., 0),
    'combined': (3., 5),
    'ood_payload': (5., 5),
    'ood_delay': (3., 8),
}


def calibration_cases():
    return [Scenario(seed, task, payload, delay)
            for seed in range(1000, 1004) for task in ('reach', 'tracking')
            for payload, delay in ((0., 0), (1.5, 2), (3., 5))]


def validation_cases():
    return [Scenario(seed, task, payload, delay)
            for seed in range(10000, 10004) for task in ('reach', 'tracking')
            for payload, delay in ((.5, 1), (2.5, 4))]


def test_cases():
    return [(name, Scenario(seed, task, payload, delay))
            for name, (payload, delay) in CONDITIONS.items()
            for seed in range(20000, 20012) for task in ('reach', 'tracking')]


def rollout(env, scenario, policy=None):
    observation = env.reset(scenario=scenario)
    total = 0.
    for _ in range(env.horizon):
        action = policy(observation) if policy else [0.] * 6
        observation, reward, done, _ = env.step(action)
        total += reward
        if done:
            break
    return {**env.metrics(), 'episode_return': total, 'episode_steps': env.steps}
