"""Behavioral checks on synthetic data only; no public-data training."""
import json
import unittest
import numpy as np

import automl_meta_research as meta


class MetaResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.classes = np.array(["zebra", "attack", "normal"])
        cls.y = cls.classes[np.arange(180) % 3]
        rng = np.random.RandomState(830)
        cls.parts = []
        for _ in range(3):
            p = rng.uniform(.01, .15, size=(180, 3))
            p[np.arange(180), np.arange(180) % 3] += .7
            cls.parts.append(p / p.sum(1, keepdims=True))
        cls.folds = [(np.flatnonzero(np.arange(180) % 3 != fold),
                      np.flatnonzero(np.arange(180) % 3 == fold)) for fold in range(3)]
        # Every fit subset must contain every class, unlike splitting on labels.
        cls.folds = [(np.flatnonzero(np.arange(180) // 3 % 3 != fold),
                      np.flatnonzero(np.arange(180) // 3 % 3 == fold)) for fold in range(3)]

    def test_probability_alignment_and_missing_class(self):
        aligned = meta.align_probabilities(np.array([[.8, .2]]),
                                           ["normal", "zebra"], self.classes)
        np.testing.assert_allclose(aligned, [[.2, 0., .8]])
        with self.assertRaises(ValueError):
            meta.align_probabilities([[1.]], ["unknown"], self.classes)

    def test_probability_payloads_are_alignment_invariant(self):
        order = [2, 0, 1]
        payloads = {name: {"probabilities": p[:, order],
                          "classes": self.classes[order]}
                    for name, p in zip(meta.COMPONENT_ORDER, self.parts)}
        state = meta.fit_normalized_state(self.parts, self.y, self.classes)
        np.testing.assert_allclose(meta.predict_state(payloads, state, self.classes),
                                   meta.predict_state(self.parts, state, self.classes))

    def test_serialized_raw_state_works_with_existing_prediction_contract(self):
        state = meta.fit_normalized_state(self.parts, self.y, self.classes)
        state = json.loads(json.dumps(state, allow_nan=False))
        p = meta.predict_state(self.parts, state, self.classes)
        np.testing.assert_allclose(p.sum(1), 1., atol=1e-12)
        self.assertGreater(np.mean(self.classes[p.argmax(1)] == self.y), .95)
        self.assertAlmostEqual(state["audit"]["mean_sample_weight"], 1.)
        self.assertTrue(state["audit"]["converged"])
        # Independently use the repository's established evaluator. This tests
        # the actual integration boundary, not a second copy of our formula.
        from automl_class_combo import predict_meta_state
        np.testing.assert_allclose(p, predict_meta_state(self.parts, state, self.classes),
                                   atol=1e-12)

    def test_unseen_targets_and_changed_class_order_rejected(self):
        bad = self.y.copy()
        bad[0] = "missing"
        with self.assertRaises(ValueError):
            meta.fit_normalized_state(self.parts, bad, self.classes)
        state = meta.fit_normalized_state(self.parts, self.y, self.classes)
        with self.assertRaises(ValueError):
            meta.predict_state(self.parts, state, self.classes[::-1])

    def test_imbalanced_weights_still_have_unit_mean(self):
        selected = np.r_[np.flatnonzero(self.y == "zebra"),
                         np.flatnonzero(self.y == "attack")[:20],
                         np.flatnonzero(self.y == "normal")[:4]]
        state = meta.fit_normalized_state([p[selected] for p in self.parts],
                                          self.y[selected], self.classes)
        audit = state["audit"]
        self.assertAlmostEqual(audit["mean_sample_weight"], 1., places=12)
        self.assertGreater(audit["class_weights"][2], audit["class_weights"][0])

    def test_binary_numeric_nonlexicographic_class_order(self):
        classes = np.array([70, -2])
        y = classes[np.arange(90) % 2]
        p = np.full((90, 2), .1)
        p[np.arange(90), np.arange(90) % 2] = .9
        state = meta.fit_normalized_state([p] * 3, y, classes)
        predicted = meta.predict_state([p] * 3, state, classes)
        np.testing.assert_array_equal(classes[predicted.argmax(1)], y)
        self.assertEqual(np.asarray(state["coef"]).shape, (1, 6))

    def test_single_class_constant_prediction(self):
        p = np.ones((8, 1))
        state = meta.fit_normalized_state([p] * 3, np.array(["only"] * 8), ["only"])
        np.testing.assert_array_equal(meta.predict_state([p] * 3, state, ["only"]), p)
        self.assertEqual(state["audit"]["fit_count"], 0)

    def test_crossfit_coverage_and_heldout_target_independence(self):
        p, audit = meta.crossfit_meta(self.parts, self.y, self.classes, self.folds)
        self.assertEqual(audit["fit_count"], 3)
        self.assertEqual(len(audit["states"]), 3)
        np.testing.assert_allclose(p.sum(1), 1.)
        altered = self.y.copy()
        valid = self.folds[0][1]
        altered[valid] = np.roll(altered[valid], 1)
        changed, _ = meta.crossfit_meta(self.parts, altered, self.classes, self.folds)
        np.testing.assert_allclose(changed[valid], p[valid], atol=1e-12)
        overlapping = [(np.arange(180), np.arange(90)), self.folds[1]]
        with self.assertRaises(ValueError):
            meta.crossfit_meta(self.parts, self.y, self.classes, overlapping)

    def test_affine_false_positive_penalty_changes_decision(self):
        p = np.array([[.5, .4, .1], [.1, .2, .7]])
        rule = {"classes": self.classes.tolist(), "a": [1., 1., 1.],
                "b": [.2, 0., 0.]}
        np.testing.assert_array_equal(meta.predict_affine_labels(p, rule),
                                      self.classes[[1, 2]])
        np.testing.assert_array_equal(meta.predict_affine(p, rule).argmax(1), [1, 2])

    def test_affine_retains_identity_if_already_perfect(self):
        p = np.eye(3)[np.arange(180) % 3]
        rule = meta.fit_affine_rule(self.y, p, self.classes)
        self.assertEqual(rule["chosen_iteration"], -1)
        self.assertEqual(rule["audit"]["macro_f1"], 1.)
        self.assertEqual(len(rule["audit"]["trajectory"]), 51)
        json.dumps(rule, allow_nan=False)

    def test_focal_mcc_finite_gradients_and_better_for_correct_prediction(self):
        import torch
        targets = torch.eye(3, dtype=torch.float64).repeat(4, 1)
        correct = (.9 * targets + .1 / 3).requires_grad_()
        wrong = torch.roll(correct.detach(), 1, dims=1).requires_grad_()
        first = meta.focal_mcc_loss(correct, targets)
        second = meta.focal_mcc_loss(wrong, targets)
        self.assertLess(float(first.detach()), float(second.detach()))
        first.backward()
        self.assertTrue(bool(torch.isfinite(correct.grad).all()))
        uniform = torch.full((12, 3), 1 / 3, dtype=torch.float64, requires_grad=True)
        value = meta.focal_mcc_loss(uniform, targets)
        value.backward()
        self.assertTrue(bool(torch.isfinite(uniform.grad).all()))

    def test_mcc_serialization_and_crossfit_reuses_initial_heads(self):
        _, initial_audit = meta.crossfit_meta(self.parts, self.y, self.classes, self.folds)
        p, audit = meta.crossfit_meta(self.parts, self.y, self.classes, self.folds,
                                      method="mcc", initial_states=initial_audit["states"])
        self.assertEqual(audit["fit_count"], 3)
        self.assertTrue(all(row["reused_normalized_state"] for row in audit["folds"]))
        self.assertTrue(np.isfinite(p).all())
        json.dumps(audit, allow_nan=False)
        state = meta.fit_normalized_state(self.parts, self.y, self.classes)
        mcc = meta.fit_mcc_state(self.parts, self.y, self.classes, state)
        self.assertEqual(mcc["audit"]["steps"], 100)
        self.assertEqual(len(mcc["audit"]["losses"]), 100)
        from automl_class_combo import predict_meta_state
        np.testing.assert_allclose(meta.predict_state(self.parts, mcc, self.classes),
                                   predict_meta_state(self.parts, mcc, self.classes), atol=1e-12)
        with self.assertRaises(ValueError):
            meta.fit_mcc_state([p[:-1] for p in self.parts], self.y[:-1], self.classes, state)


if __name__ == "__main__":
    unittest.main(verbosity=2)
