import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
from residual_control import ROOT, evaluate, report, train
from residual_control.env import Scenario, UR16eTrackingEnv


class RenderingTest(unittest.TestCase):
    def test_render_preserves_integrator_state_and_close_is_idempotent(self):
        env = UR16eTrackingEnv("controller")
        self.addCleanup(env.close)
        env.reset(scenario=Scenario(20000, "tracking", 3.0, 5))
        for _ in range(30):
            env.step(np.zeros(6))
        renderer = Mock()
        env.renderer = renderer
        fields = ["qpos", "qvel", "qacc", "qacc_warmstart", "qfrc_constraint"]
        before = {key: getattr(env.data, key).copy() for key in fields}
        env.render()
        for key, value in before.items():
            np.testing.assert_array_equal(getattr(env.data, key), value, err_msg=key)
        self.assertIsNot(renderer.render.call_args.args[0], env.data)
        env.close()
        env.close()
        renderer.close.assert_called_once()

    def test_invalid_scenarios_fail_early(self):
        for arguments in [
            {"task": "typo"},
            {"payload": -1.0},
            {"payload": np.nan},
            {"delay": -1},
            {"delay": 1.5},
        ]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                Scenario(1, **arguments)
        with self.assertRaisesRegex(ValueError, "control mode"):
            UR16eTrackingEnv("typo")


class TrainingTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        shutil.copytree(ROOT / "configs", self.root / "configs")
        (self.root / "results").mkdir()
        for name in ["controller_tuning.json", "baseline_validation.json"]:
            shutil.copyfile(ROOT / "results" / name, self.root / "results" / name)
        self.root_patch = patch.object(train, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def test_partial_run_is_not_overwritten(self):
        folder = self.root / "runs/main/residual_seed1"
        folder.mkdir(parents=True)
        sentinel = folder / "training.csv"
        sentinel.write_text("keep this partial log\n")
        with self.assertRaises(FileExistsError):
            train.main(["--method", "residual", "--seed", "1"])
        self.assertEqual(sentinel.read_text(), "keep this partial log\n")

    def test_training_budget_must_be_positive(self):
        for steps in ["0", "-1"]:
            with self.subTest(steps=steps), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    train.main(["--method", "residual", "--seed", "1", "--steps", steps])
                self.assertEqual(error.exception.code, 2)
        self.assertFalse((self.root / "runs").exists())

    def test_interrupted_validation_closes_both_environments(self):
        training_env, validation_env = Mock(), Mock()
        training_env.observation_space.shape = (87,)
        with (
            patch.object(train, "UR16eTrackingEnv", side_effect=[training_env, validation_env]),
            patch.object(train, "make_agent"),
            patch.object(train, "digest", return_value="model"),
            patch.object(train, "validation", side_effect=RuntimeError("interrupted")),
        ):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                train.main(["--method", "residual", "--seed", "1", "--steps", "1"])
        training_env.close.assert_called_once()
        validation_env.close.assert_called_once()
        metadata = json.loads((self.root / "runs/main/residual_seed1/metadata.json").read_text())
        self.assertFalse(metadata["complete"])
        self.assertFalse((self.root / "runs/main/residual_seed1/complete.json").exists())


class EvaluationTest(unittest.TestCase):
    def test_report_rejects_wrong_targets_with_correct_row_count(self):
        episodes = pd.read_csv(ROOT / "results/evaluation/controller_seed0.csv")
        report.validate_evaluation(episodes, "controller", 0)
        episodes.loc[0, "target_seed"] = 99999
        with self.assertRaisesRegex(ValueError, "held-out cases"):
            report.validate_evaluation(episodes, "controller", 0)

    def test_report_rejects_nonfinite_metrics_and_wrong_policy_labels(self):
        original = pd.read_csv(ROOT / "results/evaluation/controller_seed0.csv")
        for column, value in [
            ("tracking_rmse_m", np.nan),
            ("method", "residual"),
            ("training_seed", 1),
            ("delay_ms", 999),
        ]:
            episodes = original.copy()
            episodes.loc[0, column] = value
            with self.subTest(column=column), self.assertRaises(ValueError):
                report.validate_evaluation(episodes, "controller", 0)

    def test_policy_loader_rejects_model_and_controller_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(
                ROOT / "models", root / "models", ignore=shutil.ignore_patterns("assets")
            )
            (root / "results/learning").mkdir(parents=True)
            name = "residual_seed1.json"
            shutil.copyfile(ROOT / "results/learning" / name, root / "results/learning" / name)
            tuning = ROOT / "results/controller_tuning.json"
            shutil.copyfile(tuning, root / "results/controller_tuning.json")
            model = root / "models/ur16e.xml"
            original = model.read_bytes()
            model.write_bytes(original + b"\n")
            with patch.object(evaluate, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "Robot model differs"):
                    evaluate.load_policy("residual", 1)
                model.write_bytes(original)
                controller = json.loads(tuning.read_text())
                controller["selected"]["kp"] += 1
                (root / "results/controller_tuning.json").write_text(json.dumps(controller))
                with self.assertRaisesRegex(ValueError, "Controller differs"):
                    evaluate.load_policy("residual", 1)


if __name__ == "__main__":
    unittest.main()
