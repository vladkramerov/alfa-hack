"""H103: change only ordered-vs-plain boosting on the true OOF teacher."""
from sprint_contextual import ContextualTeacherCorrector


class OrderedHonestCorrector(ContextualTeacherCorrector):
    def __init__(self,iterations=400):
        super().__init__(iterations=iterations,depth=5,learning_rate=.05,l2_leaf_reg=10,threads=2)
        self.parameters['boosting_type']='Ordered'

    def configuration(self):
        configuration=super().configuration()
        configuration.update(hypothesis='HYP-103',family='ordered_honest_oof_contextual_catboost',
            training_teacher='H88_reused_true_grouped_raw_OOF',query_teacher='held_out_raw_xgboost',
            H48_meta_state_used=False,change_from_H88='only_boosting_type_Ordered',
            teacher_provenance='caller_must_verify_grouped_oof_and_exact_global_positions')
        return configuration
