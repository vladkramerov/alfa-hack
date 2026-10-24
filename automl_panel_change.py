"""Strictly past covariate changes and fixed H84 panel ensemble.

H83 predicts an annual first difference using only earlier training rows of
the same physical profile. H84 averages that forecast with H73 at fixed
50/50 weights. All target history and model state are learned in fit.
"""
import numpy as np
import pandas as pd
from automl_panel_memory import PanelHistory, panel_keys, PANEL_FIELDS


_IDENTIFIER_NAMES = {'id', 'row_id', 'opaque_id', 'identifier', 'sample_id'}


def _without_identifiers(X):
    # TaskData.X removes the declared identifier. This additional name guard
    # covers direct callers; it does not alter the measured public schema.
    names = [c for c in X if str(c).casefold() in _IDENTIFIER_NAMES]
    return X.drop(columns=names) if names else X


class PanelChangeForecaster:
    """H83 annual target change with train-only last-year covariates."""
    def __init__(self, threads=4):
        self.threads = threads

    def _frame(self, X):
        X = _without_identifiers(X)
        temporal, matched = self.history_.features(X)
        keys = panel_keys(X, PANEL_FIELDS)
        years = X.Year_Factor.to_numpy(float)
        last = np.full((len(X), len(self.numeric_)), np.nan)
        gaps = np.ones(len(X))
        for i, (key, year) in enumerate(zip(keys, years)):
            record = self.past_x_.get(key)
            if record is None:
                continue
            past = record[record[:, 0] < year]
            if not len(past):
                continue
            latest = past[-1]
            gaps[i] = year-latest[0]
            last[i] = latest[1:]
        current = X[self.numeric_].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        extra = pd.DataFrame(np.column_stack([last, (current-last)/gaps[:, None]]),
            index=X.index,
            columns=['__last_x_'+c for c in self.numeric_]+['__annual_dx_'+c for c in self.numeric_])
        frame = pd.concat([X.drop(columns=['Year_Factor']), temporal, extra], axis=1)
        if not frame.columns.is_unique:
            raise ValueError('Panel change feature collision')
        return frame, temporal, matched, gaps

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        from automl_models import Encoder
        from automl_combo_memory import fit_once
        dropped = [c for c in X if str(c).casefold() in _IDENTIFIER_NAMES]
        X = _without_identifiers(X)
        self.history_ = PanelHistory().fit(X, y)
        self.numeric_ = [c for c in X.select_dtypes(include='number') if c != 'Year_Factor']
        data = X[self.numeric_].replace([np.inf, -np.inf], np.nan).copy()
        data['__key'] = panel_keys(X, PANEL_FIELDS)
        data['__year'] = X.Year_Factor.to_numpy(float)
        moments = data.groupby(['__key', '__year'], sort=False)[self.numeric_].mean().reset_index()
        self.past_x_ = {key:part.sort_values('__year')[['__year']+self.numeric_].to_numpy(float)
                        for key, part in moments.groupby('__key', sort=False)}
        frame, temporal, matched, gaps = self._frame(X)
        if matched.sum() < 500:
            raise ValueError('Insufficient strictly causal covariate pairs')
        self.encoder_ = Encoder(native=True, scale=False)
        values = self.encoder_.fit_transform(frame.loc[matched])
        target = (np.asarray(y, float)[matched]-temporal.loc[matched, '__panel_last'].to_numpy())/gaps[matched]
        settings = dict(n_estimators=600, learning_rate=.03, num_leaves=15,
                        min_child_samples=40, reg_lambda=10., verbosity=-1,
                        random_state=42, n_jobs=self.threads, force_col_wise=True)
        self.model_ = fit_once(LGBMRegressor(**settings), values, target,
            {'strategy':'causal_covariate_first_differences', 'parameters':settings,
             'columns':list(frame.columns), 'target':'annual_current_minus_last_strict_past_year_mean'})
        self.nonnegative_ = bool((np.asarray(y) >= 0).all())
        self.audit_ = {**self.history_.audit_, 'strategy':'causal_covariate_first_differences',
            'causal_rows':int(matched.sum()), 'numeric_past_and_delta_fields':self.numeric_,
            'parameters':settings, 'validation_targets_used':False, 'year_gap_scaling':True,
            'id_feature_used':False, 'dropped_identifier_columns':dropped}
        return self

    def predict_with_fallback(self, X, fallback):
        frame, temporal, matched, gaps = self._frame(X)
        prediction = np.asarray(fallback, float).copy()
        if prediction.shape != (len(X),) or not np.isfinite(prediction).all():
            raise ValueError('Invalid fallback shape or values')
        if matched.any():
            prediction[matched] = temporal.loc[matched, '__panel_last'].to_numpy()+gaps[matched]*self.model_.predict(
                self.encoder_.transform(frame.loc[matched]))
            if self.nonnegative_:
                prediction[matched] = np.maximum(0, prediction[matched])
        if not np.isfinite(prediction).all():
            raise ValueError('Nonfinite panel change prediction')
        return prediction


class PanelChangeHybridModel:
    """H84 fixed average of H73 and the H83 covariate-change forecaster."""
    def __init__(self, candidate=None, threads=4):
        self.candidate, self.threads = dict(candidate or {}), threads

    def fit(self, X, y):
        from automl_panel_memory import PanelHybridModel
        X = _without_identifiers(X)
        self.panel_ = PanelHybridModel({'family':'extra_trees', 'parameters':{},
                                       'strategy':'panel_hybrid_memory'}, self.threads).fit(X, y)
        self.base_ = self.panel_.base_
        self.change_ = PanelChangeForecaster(self.threads).fit(X, y)
        self.keep_mask_ = np.ones(len(X), dtype=bool)
        self.removed_positions_ = np.empty(0, dtype=int)
        self.audit_ = {'strategy':'panel_change_hybrid', 'fit_original_rows':len(X),
            'fit_retained_rows':len(X), 'row_deletion':False, 'fixed_blend_weights':[.5, .5],
            'id_feature_used':False, 'validation_targets_used':False,
            'panel':self.panel_.audit_, 'change':self.change_.audit_}
        return self

    def predict(self, X):
        X = _without_identifiers(X)
        prior_panel = np.asarray(self.panel_.predict(X), float)
        fallback = self.base_.predict(X)
        change = self.change_.predict_with_fallback(X, fallback)
        return .5*prior_panel+.5*change
