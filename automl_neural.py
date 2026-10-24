"""Task-local temporal calibration of a complementary, unpretrained neural model."""
import numpy as np
import pandas as pd


class NeuralComplement:
    def __init__(self, candidate, threads=4):
        self.candidate, self.threads = candidate, threads

    def fit(self, X, y):
        from automl_models import build_pipeline
        from automl_torch_tabm import TorchTabM
        target = np.asarray(y, float)
        column = self.candidate['panel_time_column']
        periods = pd.to_numeric(X[column], errors='raise').to_numpy(float)
        if not np.isfinite(periods).all():
            raise ValueError('Finite temporal calibration required')
        base_candidate = {k:v for k,v in self.candidate.items() if k!='neural_complement'}
        self.weight_, self.neural_ = 0., None
        calibration = {'status':'insufficient_temporal_training', 'public_fitted_state':False}
        if len(np.unique(periods)) >= 3:
            fit, valid = np.flatnonzero(periods<periods.max()), np.flatnonzero(periods==periods.max())
            if len(fit)>=1000 and len(valid)>=100:
                base = build_pipeline('regression',base_candidate,self.threads).fit(X.iloc[fit],target[fit])
                neural = TorchTabM(kind='regression',seconds=120,threads=min(4,self.threads)).fit(X.iloc[fit],target[fit])
                baseline = np.asarray(base.predict(X.iloc[valid]),float)
                delta = neural.predict(X.iloc[valid])-baseline
                weight = float(np.clip(delta@(target[valid]-baseline)/max(delta@delta,1e-12),0,1))
                before = float(np.sqrt(np.mean((target[valid]-baseline)**2)))
                after = float(np.sqrt(np.mean((target[valid]-baseline-weight*delta)**2)))
                if weight>=.05 and after<=before*.995:
                    self.weight_ = weight
                calibration = {'status':'accepted' if self.weight_ else 'rejected_inner',
                    'fit_rows':len(fit),'valid_rows':len(valid),'maximum_prefix_period':float(periods[fit].max()),
                    'calibration_period':float(periods.max()),'base_rmse':before,'blend_rmse':after,
                    'weight':self.weight_,'query_targets_used':False,'public_fitted_state':False,
                    'neural_inner':neural.audit_}
                del base, neural
        self.base_ = build_pipeline('regression',base_candidate,self.threads).fit(X,target)
        if self.weight_:
            self.neural_ = TorchTabM(kind='regression',seconds=120,threads=min(4,self.threads)).fit(X,target)
        self.feature_code_ = self.base_.feature_code_
        self.classes_ = None
        self.audit_ = {**self.base_.audit_,'neural_complement':calibration,
                       'neural_full':self.neural_.audit_ if self.neural_ is not None else None}
        return self

    def predict(self, X):
        base = self.base_.predict(X)
        return base if self.neural_ is None else (1-self.weight_)*base+self.weight_*self.neural_.predict(X)
