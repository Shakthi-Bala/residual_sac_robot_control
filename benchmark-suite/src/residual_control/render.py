import argparse
import json
from contextlib import ExitStack

import numpy as np
import torch
from infrastructure import pytorch_util as ptu
from PIL import Image
from scripts.gif_style import comparison_frame

from residual_control import ROOT
from residual_control.artifacts import save_json
from residual_control.env import Scenario, UR16eTrackingEnv
from residual_control.evaluate import load_policy
from residual_control.protocol import CONDITIONS

DEMOS = [
    ("nominal_reach", "nominal", "reach", "best"),
    ("combined_reach", "combined", "reach", "best"),
    ("combined_tracking", "combined", "tracking", "best"),
    ("ood_payload_tracking", "ood_payload", "tracking", "best"),
    ("ood_delay_tracking", "ood_delay", "tracking", "best"),
    ("final_tracking", "combined", "tracking", "final"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(1)
    ptu.init_gpu(use_gpu=False)
    methods = ["controller", "standalone", "residual"]
    labels = ["Cartesian PI", "Standalone SAC", "Residual SAC"]
    controller = json.loads((ROOT / "results/controller_tuning.json").read_text())["selected"]
    policies, checkpoint_steps = {}, {}
    for checkpoint in ("best", "final"):
        policies[checkpoint] = [None]
        checkpoint_steps[checkpoint] = {}
        for method in methods[1:]:
            agent, metadata = load_policy(method, args.seed, checkpoint)
            policies[checkpoint].append(agent)
            step_key = "best_validation_step" if checkpoint == "best" else "training_interactions"
            checkpoint_steps[checkpoint][method] = metadata[step_key]
    environments = []
    with ExitStack() as stack:
        for method in methods:
            env = UR16eTrackingEnv(method, presentation=True, **controller)
            stack.callback(env.close)
            environments.append(env)
        folder = ROOT / "results/media"
        folder.mkdir(parents=True, exist_ok=True)
        for name, condition, task, checkpoint in DEMOS:
            agents = policies[checkpoint]
            payload, delay = CONDITIONS[condition]
            case = Scenario(20000, task, payload, delay)
            observations = [env.reset(scenario=case) for env in environments]
            frames, finished = [], [False] * 3
            for step in range(environments[0].horizon):
                for i, (env, agent) in enumerate(zip(environments, agents)):
                    if not finished[i]:
                        action = (
                            agent.deterministic_action(observations[i]) if agent else np.zeros(6)
                        )
                        observations[i], _, finished[i], _ = env.step(action)
                if step % 5:
                    continue
                canvas = comparison_frame(
                    [Image.fromarray(env.render()) for env in environments],
                    labels,
                    [env.trace[-1]["error"] * 1000 for env in environments],
                    [bool(env.failures) for env in environments],
                    task,
                    checkpoint,
                    payload,
                    delay * 20,
                    (step + 1) * 0.02,
                )
                frames.append(canvas)

            path = folder / f"{name}.gif"
            temporary = path.with_suffix(".tmp.gif")
            frames[0].save(
                temporary,
                save_all=True,
                append_images=frames[1:],
                duration=100,
                loop=0,
                optimize=False,
            )
            temporary.replace(path)
            save_json(
                path.with_suffix(".json"),
                {
                    "target_seed": 20000,
                    "training_seed": args.seed,
                    "selection": f"fixed training seed {args.seed} and target 20000; best validation or final checkpoint as labeled; not selected for test performance",
                    "checkpoint": checkpoint,
                    "checkpoint_steps": checkpoint_steps[checkpoint],
                    "task": task,
                    "condition": condition,
                    "payload_kg": payload,
                    "delay_ms": delay * 20,
                    "presentation": "light comparison cards, navy headers, per-frame position error",
                    "playback": "10 fps, real time",
                    "metrics": {m: e.metrics() for m, e in zip(methods, environments)},
                },
            )
            print(path, flush=True)


if __name__ == "__main__":
    main()
