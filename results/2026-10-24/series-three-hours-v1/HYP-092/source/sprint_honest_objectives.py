"""H92--94 unchanged residual objectives paired with honest RAW OOF teachers."""
from sprint_ambitious import LovaszTeacherBoost
from sprint_classification import TeacherResidualF1Classifier
from sprint_softboost import MacroTeacherBoost


def _provenance(number, family):
    return dict(hypothesis=f'HYP-{number:03d}',family=family,
        training_teacher='reused_H88_nested_grouped_oof_raw_xgboost',
        query_teacher='original_held_out_raw_xgboost', H48_meta_state_used=False,
        new_teacher_fits=0, teacher_provenance='caller_must_verify_reused_honest_teacher_artifacts')


class HonestMacroTeacherBoost(MacroTeacherBoost):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def configuration(self):
        result=super().configuration()
        result.update(_provenance(92,'honest_oof_soft_macro_f1_lightgbm'))
        return result


class HonestLovaszTeacherBoost(LovaszTeacherBoost):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)

    def configuration(self):
        result=super().configuration()
        result.update(_provenance(93,'honest_oof_lovasz_lightgbm'))
        return result


class HonestResidualF1Classifier(TeacherResidualF1Classifier):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        # Parent intentionally exposes a dictionary property, not a method.
        self.configuration.update(_provenance(94,'honest_oof_soft_macro_f1_numpy_mlp'))
