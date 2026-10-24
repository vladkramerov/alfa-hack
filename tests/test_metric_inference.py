"""Metric contract checks on synthetic text and predictions; no model fits."""
import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from sklearn import metrics

import automl_data as api


class MetricInferenceTests(unittest.TestCase):
    def infer(self, text, default="rmse"):
        self.assertTrue(hasattr(api, "infer_metric"), "explicit metric inference is not implemented")
        return api.infer_metric(text, default=default)

    def test_primary_line_overrides_secondary_mentions_and_examples(self):
        self.assertEqual(self.infer("Regression. Main metric: MAE.\nExample RMSE calculation below."), "mae")
        self.assertEqual(self.infer("Metric: MAE\nAn example reports RMSE."), "mae")
        self.assertEqual(self.infer("Primary metric: MAE; secondary metric: RMSE"), "mae")

    def test_metric_heading_takes_following_value(self):
        self.assertEqual(self.infer("## Metric\n\n**MAE**\n\n## Example\nRMSE"), "mae")
        self.assertEqual(self.infer("## Main metric\nMAE\n\nSecondary metric: RMSE"), "mae")

    def test_russian_primary_and_public_style_lines(self):
        self.assertEqual(self.infer("Основная метрика — **Macro F1**.\nДополнительно Accuracy и Weighted F1."), "macro_f1")
        self.assertEqual(self.infer("Метрика оценки — **RMSE** на исходной шкале."), "rmse")
        self.assertEqual(self.infer("Основная метрика: MAE.\nПример RMSE."), "mae")

    def test_main_declaration_has_priority_over_generic_declaration(self):
        self.assertEqual(self.infer("Metric: RMSE\nPrimary metric: MAE"), "mae")

    def test_specific_aliases_do_not_also_match_generic_metric(self):
        aliases = {"weighted F1": "weighted_f1", "F1 weighted": "weighted_f1",
                   "weighted_f1": "weighted_f1", "F1 macro": "macro_f1",
                   "balanced accuracy": "balanced_accuracy", "balanced_accuracy": "balanced_accuracy",
                   "ROC-AUC": "roc_auc", "Log Loss": "log_loss", "R²": "r2",
                   "MSE": "mse", "RMSLE": "rmsle", "MAPE": "mape", "Accuracy": "accuracy"}
        for alias, expected in aliases.items():
            with self.subTest(alias=alias):
                self.assertEqual(self.infer("Metric: " + alias), expected)

    def test_ambiguous_explicit_and_unlabelled_metrics_raise(self):
        for text in ["Primary metric: MAE or RMSE", "Main metric: MAE\nPrimary metric: RMSE",
                     "We evaluate MAE and RMSE."]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "[Aa]mbiguous"):
                self.infer(text)

    def test_unsupported_explicit_metric_never_silently_defaults(self):
        for text in ["Metric: SMAPE", "Primary metric: NDCG.\nExample: RMSE", "## Metric\nCohen kappa",
                     "Primary metric: SMAPE (RMSE is only a comparison)"]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "[Uu]nsupported"):
                self.infer(text)

    def test_sole_mention_and_no_mention_defaults(self):
        self.assertEqual(self.infer("Minimise the MAE of predictions."), "mae")
        self.assertEqual(self.infer("Predict a real-valued output."), "rmse")
        self.assertEqual(self.infer("Predict labels.", default="macro_f1"), "macro_f1")

    def test_load_task_preserves_fixture_contract_and_metric_determines_kind(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / "tmp") as root:
            folder = Path(root)
            pd.DataFrame({"key": range(6), "x": range(6), "outcome": [0, 1] * 3}).to_csv(folder / "train.csv", index=False)
            pd.DataFrame({"key": [9, 10], "x": [9, 10]}).to_csv(folder / "test.csv", index=False)
            pd.DataFrame({"key": [9, 10], "outcome": [0, 0]}).to_csv(folder / "sample_submission.csv", index=False)
            pd.DataFrame({"feature": ["key", "x", "outcome"], "role": ["id", "feature", "target"]}).to_csv(folder / "data_dictionary.csv", index=False)
            task = folder / "task.md"
            task.write_text("Regression is a different example. Primary metric: balanced accuracy.\ntrain.csv test.csv sample_submission.csv data_dictionary.csv", encoding="utf-8")
            bundle = api.load_task(task)
            self.assertEqual((bundle.target, bundle.id_column, bundle.task_type, bundle.metric),
                             ("outcome", "key", "classification", "balanced_accuracy"))
            self.assertEqual(list(bundle.X), ["x"])


class ExtendedScoreTests(unittest.TestCase):
    def test_mse_mape_weighted_f1_and_balanced_accuracy_use_declared_metric(self):
        y, prediction = np.array([1., 4., 8.]), np.array([2., 2., 9.])
        self.assertAlmostEqual(api.score("mse", y, prediction), metrics.mean_squared_error(y, prediction))
        self.assertAlmostEqual(api.score("mape", y, prediction), metrics.mean_absolute_percentage_error(y, prediction))
        actual, labels = ["a", "a", "a", "b"], ["a", "a", "b", "b"]
        self.assertAlmostEqual(api.score("weighted_f1", actual, labels), metrics.f1_score(actual, labels, average="weighted"))
        self.assertAlmostEqual(api.score("balanced_accuracy", actual, labels), metrics.balanced_accuracy_score(actual, labels))

    def test_rmsle_clips_predictions_to_zero_but_rejects_negative_actual(self):
        self.assertAlmostEqual(api.score("rmsle", [0., 3.], [-2., 3.]), 0.)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            api.score("rmsle", [-1., 3.], [0., 3.])

    def test_all_metric_directions_and_unknown_metric_error(self):
        self.assertTrue(hasattr(api, "LOWER_IS_BETTER"), "shared metric direction set is missing")
        self.assertEqual(api.LOWER_IS_BETTER, {"rmse", "mae", "mse", "rmsle", "mape", "log_loss"})
        for metric in api.LOWER_IS_BETTER:
            self.assertTrue(api.better(metric, .1, .2)); self.assertFalse(api.better(metric, .2, .1))
        for metric in ["r2", "macro_f1", "weighted_f1", "balanced_accuracy", "accuracy", "roc_auc"]:
            self.assertTrue(api.better(metric, .2, .1)); self.assertFalse(api.better(metric, .1, .2))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            api.score("unknown_metric", [0, 1], [0, 1])


if __name__ == "__main__":
    unittest.main()
