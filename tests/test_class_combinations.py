import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from automl_class_combo import (ClassBranch, FrequencyFeatureTransformer, aligned_probabilities,
    bounded_smote, stack_features, fit_meta_state, predict_meta_state, build_class_combo)


class ClassCombinationTests(unittest.TestCase):
    def test_alignment_and_stack_clip(self):
        p = aligned_probabilities([[.2, .8]], ['b', 'a'], ['a', 'b'])
        np.testing.assert_allclose(p, [[.8, .2]])
        features = stack_features([np.array([[0., 1.]])]*3)
        self.assertEqual(features.shape, (1, 6))
        self.assertTrue(np.isfinite(features).all())
        with self.assertRaises(ValueError):
            aligned_probabilities([[np.nan]], ['a'], ['a'])

    def test_frequency_missing_literal_and_unseen(self):
        X = pd.DataFrame({'proto': ['M:', None, 'M:'], 'service': ['a|b', 'a', 'a|b'], 'state': ['x']*3})
        t = FrequencyFeatureTransformer().fit(X)
        out = t.transform(X)
        self.assertEqual(out['proto__frequency'].tolist(), [2/3, 1/3, 2/3])
        new = X.iloc[:1].copy(); new['proto'] = 'new'
        self.assertEqual(t.transform(new)['proto__frequency'].iloc[0], 0.)

    def test_smote_preserves_raw_rows_and_constraints(self):
        X = pd.DataFrame({'proto': ['udp']*18, 'service': ['dns']*18, 'state': ['ok']*18,
                          'spkts': np.arange(18), 'stcpb': np.zeros(18)})
        y = np.array(['a']*6+['b']*12)
        sampled, target, audit, synthetic = bounded_smote(X, y)
        pd.testing.assert_frame_equal(sampled.iloc[:len(X)].reset_index(drop=True), X, check_dtype=False)
        self.assertEqual(len(target), len(sampled))
        self.assertEqual(audit['generated_accepted'], len(synthetic))
        np.testing.assert_allclose(synthetic['spkts'], np.rint(synthetic['spkts']))
        np.testing.assert_allclose(synthetic['stcpb'], 0.)

    def test_frozen_meta_json_roundtrip_binary_and_single(self):
        y = np.array(['a','b']*6)
        p = np.tile([[.8,.2],[.2,.8]], (6,1))
        state, audit = fit_meta_state([p,p,p], y, np.array(['a','b']))
        state = json.loads(json.dumps(state))
        result = predict_meta_state([p,p,p], state, np.array(['a','b']))
        self.assertEqual(result.shape, p.shape)
        np.testing.assert_allclose(result.sum(axis=1), 1)
        self.assertEqual(audit['class_weight'], {'a': 1/np.sqrt(6), 'b': 1/np.sqrt(6)})
        one, _ = fit_meta_state([np.ones((3,1))]*3, np.array(['a']*3), np.array(['a']))
        np.testing.assert_allclose(predict_meta_state([np.ones((2,1))]*3, one, ['a']), 1.)

    def test_frozen_rule_skips_oof_and_handles_unseen(self):
        X = pd.DataFrame({'proto': ['tcp','udp']*8, 'service': ['a']*16, 'state': ['ok']*16, 'value': range(16)})
        candidate = {'strategy':'frequency_blend_macro_f1', 'parameters':{'n_estimators':3},
                     'class_multipliers': {'a':1.,'b':1.25}}
        with patch('automl_class_combo.calibration_folds', side_effect=AssertionError('OOF forbidden')):
            model = build_class_combo(candidate, threads=1).fit(X, np.array(['a','b']*8))
        new = X.iloc[:3].copy(); new['proto'] = 'new'
        p = model.predict_proba(new)
        self.assertEqual(p.shape, (3,2))
        np.testing.assert_allclose(p.sum(axis=1), 1.)
        self.assertEqual(model.audit_['state_source'], 'frozen_candidate')
        self.assertEqual(model.removed_positions_.size, 0)

    def test_single_class_uses_no_underlying_estimator(self):
        X = pd.DataFrame({'proto':['tcp']*5, 'service':['a']*5, 'state':['x']*5, 'n':range(5)})
        with patch('automl_class_combo._fit_once', side_effect=AssertionError('No fit needed')):
            model = build_class_combo({'strategy':'probability_stacking'}).fit(X, np.array(['only']*5))
        self.assertEqual(model.predict(X).tolist(), ['only']*5)
        np.testing.assert_allclose(model.predict_proba(X), 1.)

    def test_smote_ineligible_matches_raw_encoder_fallback(self):
        X=pd.DataFrame({'proto':['tcp','udp']*5,'n':range(10)})
        branch=ClassBranch('smote',{'n_estimators':3},threads=1).fit(X,np.array(['a','b']*5))
        self.assertFalse(branch.encoder_.scale)
        self.assertEqual(branch.audit_['configuration']['policy'],'legacy')

    def test_frozen_stack_skips_oof_and_meta_fit(self):
        X = pd.DataFrame({'proto':['tcp','udp']*8,'service':['a']*16,'state':['x']*16,'n':range(16)})
        state = {'classes':['a','b'],'component_order':['raw','smote','frequency'],
                 'coef':[[0.]*6],'intercept':[0.],'clip':1e-6}
        candidate = {'strategy':'probability_stacking','parameters':{'n_estimators':3},'meta_state':state}
        with patch('automl_class_combo.calibration_folds',side_effect=AssertionError('OOF forbidden')), \
                patch('automl_class_combo.fit_meta_state',side_effect=AssertionError('Meta fit forbidden')):
            model = build_class_combo(candidate,threads=1).fit(X,np.array(['a','b']*8))
        np.testing.assert_allclose(model.predict_proba(X),.5)

    def test_offline_cache_reuse_and_legacy_missing_never_refits(self):
        from experiments.class_combinations import evaluate_class
        from experiments.twenty_advanced import inner_fingerprint,inner_folds
        X = pd.DataFrame({'proto':['tcp','udp']*12,'service':['a']*24,'state':['x']*24,'n':range(24)})
        y = np.array(['a','b']*12); positions = np.arange(24)+100
        probabilities = np.tile([.4,.6],(24,1)); valid = X.iloc[:4]
        calls = []
        class FakeBranch:
            def __init__(self,component,*args): self.component=component
            def fit(self,Xfit,yfit):
                calls.append(self.component); self.classes_=np.array(['a','b'])
                self.audit_={'generated_accepted':0,'accepted':[],'generated_target':[]}
                self.synthetic_=Xfit.iloc[:0].copy(); return self
            def predict_proba(self,Xnew): return np.tile([.4,.6],(len(Xnew),1))
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]/'tmp') as folder:
            root = Path(folder); legacy = root/'legacy'; legacy.mkdir()
            context = {'output':root/'output','cache_dir':root/'cache','legacy_cache_dir':legacy,
                       'fit_positions':positions,'valid_positions':np.arange(4)+1000,
                       'outer_components':{c:{'classes':['a','b'],'probabilities':probabilities[:4]} for c in ['raw','smote','frequency']}}
            with patch('experiments.class_combinations.ClassBranch',FakeBranch), patch('experiments.class_combinations.joblib.dump'):
                with self.assertRaises(FileNotFoundError): evaluate_class(47,X,y,valid,context)
                self.assertEqual(calls,[])
                key=inner_fingerprint('multiclass',X,y,positions,{'policy':'legacy'})
                directory=legacy/key; directory.mkdir()
                for number,(fit,held) in enumerate(inner_folds('multiclass',X,y)):
                    np.savez(directory/f'fold-{number}.npz',fit=fit,valid=held,fit_positions=positions[fit],
                             valid_positions=positions[held],classes=np.array(['a','b']),probabilities=probabilities[held])
                result46=evaluate_class(46,X,y,valid,context)
                result47=evaluate_class(47,X,y,valid,context)
                result48=evaluate_class(48,X,y,valid,context)
                self.assertEqual(calls,['smote']*3+['frequency']*3)
                self.assertTrue(all(e['status']=='reused' for c in result48['audit']['components'].values() for e in c['folds']))
                for result in [result46,result47,result48]:
                    self.assertEqual(result['audit']['outer_model_fits'],0)
                    np.testing.assert_allclose(result['probabilities'].sum(axis=1),1.)


if __name__ == '__main__': unittest.main()
