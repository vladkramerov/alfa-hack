"""H88: unchanged H74 corrector, expecting honest grouped-OOF RAW teachers."""
from sprint_contextual import ContextualTeacherCorrector


class HonestContextCorrector(ContextualTeacherCorrector):
    def __init__(self):
        super().__init__(iterations=400, depth=5, learning_rate=.05, l2_leaf_reg=10, threads=2)

    def configuration(self):
        configuration = super().configuration()
        configuration.update(hypothesis='HYP-088', family='honest_raw_oof_contextual_catboost',
            training_teacher='grouped_oof_raw_xgboost', query_teacher='held_out_raw_xgboost',
            H48_meta_state_used=False,
            teacher_provenance='caller_must_verify_grouped_oof_and_exact_global_positions')
        return configuration

    def fit(self, X, y, teacher_probabilities):
        super().fit(X, y, teacher_probabilities)
        self.audit_.update(expected_training_teacher='grouped_oof_raw_xgboost',
                           teacher_oof_provenance_is_callers_responsibility=True,
                           H48_meta_state_used=False)
        return self
