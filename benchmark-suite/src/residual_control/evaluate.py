import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from infrastructure import pytorch_util as ptu

from residual_control import ROOT
from residual_control.agent import make_agent
from residual_control.artifacts import digest, save_json
from residual_control.env import UR16eTrackingEnv
from residual_control.protocol import rollout, test_cases


def load_policy(method, seed, checkpoint="best"):
    if checkpoint not in ("best", "final"):
        raise ValueError(f"Unknown checkpoint: {checkpoint}")
    name = f"{method}_seed{seed}"
    folder = ROOT / "runs/main" / name
    weights = folder / f"{checkpoint}.pt"
    metadata_path = (
        folder / "metadata.json" if weights.exists() else ROOT / "results/learning" / f"{name}.json"
    )
    metadata = json.loads(metadata_path.read_text())
    if not metadata["complete"]:
        raise RuntimeError(f"Incomplete training: {metadata_path}")
    if metadata["model_sha256"] != digest(ROOT / "models/ur16e.xml"):
        raise ValueError("Robot model differs from the training model")
    controller = json.loads((ROOT / "results/controller_tuning.json").read_text())["selected"]
    if metadata["controller"] != controller:
        raise ValueError("Controller differs from the frozen training controller")

    if weights.exists():
        state = torch.load(weights, map_location="cpu", weights_only=True)
    else:
        archive = ROOT / "results/policies" / f"{name}_{checkpoint}.npz"
        with np.load(archive, allow_pickle=False) as arrays:
            state = {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}
    agent = make_agent(UR16eTrackingEnv.observation_shape, metadata["config"])
    agent.load_state_dict(state)
    agent.eval()
    return agent, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["controller", "standalone", "residual"], required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = 0 if args.method == "controller" else 1
    if args.method == "controller" and args.seed != 0:
        parser.error("the controller has no training seed; use --seed 0")
    torch.set_num_threads(1)
    ptu.init_gpu(use_gpu=False)
    controller = json.loads((ROOT / "results/controller_tuning.json").read_text())["selected"]
    agent, metadata = (
        load_policy(args.method, args.seed) if args.method != "controller" else (None, {})
    )
    env = UR16eTrackingEnv(args.method, **controller)
    rows = []
    try:
        for condition, scenario in test_cases():
            row = {
                "method": args.method,
                "training_seed": args.seed,
                "condition": condition,
                "task": scenario.task,
                "target_seed": scenario.seed,
                "payload_kg": scenario.payload,
                "delay_ms": scenario.delay * 20,
                **rollout(env, scenario, agent.deterministic_action if agent else None),
            }
            rows.append(row)
    finally:
        env.close()
    folder = ROOT / "results/evaluation"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{args.method}_seed{args.seed}.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    save_json(
        path.with_suffix(".json"),
        {
            "method": args.method,
            "training_seed": args.seed,
            "checkpoint_step": metadata.get("best_validation_step"),
            "training_revision": metadata.get("source_revision"),
            "model_sha256": digest(ROOT / "models/ur16e.xml"),
            "evaluation_source_sha256": digest(__file__),
            "protocol_source_sha256": digest(Path(__file__).with_name("protocol.py")),
            "test_interactions": sum(r["episode_steps"] for r in rows),
            "episodes": len(rows),
            "selection": "best validation checkpoint; test targets never select weights",
            "mean_metrics": {
                k: float(np.mean([r[k] for r in rows]))
                for k in ("success", "tracking_rmse_m", "final_error_m", "failure")
            },
        },
    )
    print(path, flush=True)


if __name__ == "__main__":
    main()
