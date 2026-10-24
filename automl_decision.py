"""Task-local posterior calibration and expected Macro-F1 decision optimization."""
import numpy as np
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize


class PosteriorDecision:
    def fit(self, probabilities, target):
        p = np.asarray(probabilities, float)
        target = np.asarray(target, int)
        logits = np.log(np.clip(p, 1e-6, 1))
        def objective(theta):
            score = logits * np.exp(-theta[0]) + theta[1:]
            loss = np.mean(logsumexp(score, axis=1) - score[np.arange(len(p)), target]) + .005 * (theta @ theta)
            gradient = softmax(score, axis=1)
            gradient[np.arange(len(p)), target] -= 1
            gradient /= len(p)
            return loss, np.r_[np.sum(gradient * (-logits * np.exp(-theta[0]))), gradient.sum(axis=0)] + .01 * theta
        initial = np.zeros(p.shape[1] + 1)
        result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=[(-2.3, 2.3)] + [(-5, 5)] * p.shape[1],
                          options={'maxiter': 100, 'ftol': 1e-10})
        self.theta_ = result.x if np.isfinite(result.fun) and result.fun <= objective(initial)[0] else initial
        p = self.calibrate(p)
        prior = np.bincount(target, minlength=p.shape[1]) / len(p)
        gain, offset = np.ones(p.shape[1]), np.zeros(p.shape[1])
        best = -np.inf
        for _ in range(26):
            decision = np.argmax(p * gain + offset, axis=1)
            mass = np.bincount(decision, minlength=p.shape[1]) / len(p)
            true = np.bincount(decision, weights=p[np.arange(len(p)), decision], minlength=p.shape[1]) / len(p)
            denominator = prior + mass + 1e-8
            value = np.mean(2 * true / denominator)
            if value > best:
                self.gain_, self.offset_, best = gain.copy(), offset.copy(), value
            gain = .5 * gain + 1 / denominator
            offset = .5 * offset - true / denominator ** 2
        self.audit_ = {'initial_ce': objective(initial)[0], 'selected_ce': objective(self.theta_)[0],
                       'optimizer_iterations': int(result.nit), 'expected_f1': float(best),
                       'teacher_oof': False, 'validation_targets_used': False}
        return self

    def calibrate(self, probabilities):
        return softmax(np.log(np.clip(probabilities, 1e-6, 1)) * np.exp(-self.theta_[0]) + self.theta_[1:], axis=1)

    def predict_proba(self, probabilities):
        return softmax(self.calibrate(probabilities) * self.gain_ + self.offset_, axis=1)
