"""SPEC-009 classification contracts; derivatives and synthetic data only."""
import unittest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

from automl_next_class import (softmax, soft_derivatives, soft_cross_entropy,
    mix_soft_targets, RareCategoryCap, masked_copy, median_rbf_gamma,
    build_next_class)


def toy(rows=24):
    return pd.DataFrame({'proto':['tcp','udp']*(rows//2),'service':['http']*rows,
                         'state':['ok']*rows,'bytes':np.arange(rows,dtype=float),
                         'packets':np.arange(rows)%5})


class NextClassificationTests(unittest.TestCase):
    def test_soft_gradient_matches_finite_difference_and_hessian_bound(self):
        z=np.array([[.2,-.7,1.1],[-2.,.4,.3]])
        q=np.array([[.2,.6,.2],[.7,.1,.2]])
        shift=np.log([9.,3.,1.])
        gradient,bound=soft_derivatives(z,q,shift)
        eps=1e-5; numeric=np.zeros_like(z)
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                delta=np.zeros_like(z);delta[i,j]=eps
                numeric[i,j]=(soft_cross_entropy(z+delta,q,shift)-soft_cross_entropy(z-delta,q,shift))/(2*eps)
        np.testing.assert_allclose(gradient,numeric,rtol=1e-7,atol=1e-8)
        for i in range(len(z)):
            numerical_hessian=np.empty((3,3))
            for j in range(3):
                delta=np.zeros_like(z);delta[i,j]=eps
                numerical_hessian[:,j]=(soft_derivatives(z+delta,q,shift)[0][i]-soft_derivatives(z-delta,q,shift)[0][i])/(2*eps)
            self.assertGreaterEqual(np.linalg.eigvalsh(np.diag(bound[i])-numerical_hessian).min(),-1e-9)
        self.assertTrue(np.all(bound>0))

    def test_extreme_logits_and_teacher_mixture(self):
        p=softmax([[1000.,-1000.,0.]])
        np.testing.assert_allclose(p.sum(axis=1),1.)
        q=mix_soft_targets(np.array([0,1]),np.array([[.2,.8],[.6,.4]]),2)
        np.testing.assert_allclose(q,[[.76,.24],[.18,.82]])
        gradient,hessian=soft_derivatives(np.array([[1e6,-1e6]]),np.array([[.5,.5]]))
        self.assertTrue(np.isfinite(gradient).all() and np.isfinite(hessian).all())
        with self.assertRaises(ValueError):mix_soft_targets(np.array([0]),[[np.nan,1]],2)

    def test_rare_cap_fit_only_and_unseen(self):
        X=toy(300);X['service']=['category'+str(i) for i in range(300)]
        cap=RareCategoryCap(128).fit(X)
        out=cap.transform(X)
        self.assertLessEqual(out.service.nunique(dropna=False),128)
        seen=cap.levels_['service'].copy()
        unseen=X.iloc[:1].copy();unseen['service']='brandnew'
        result=cap.transform(unseen)
        self.assertEqual(cap.levels_['service'],seen)
        self.assertEqual(result.service.iloc[0],'R:')
        pd.testing.assert_series_equal(out['bytes'],X['bytes'])

    def test_masked_copy_preserves_originals_keys_and_order(self):
        X=toy(200);X['proto']=np.arange(200);X['Year_Factor']=np.arange(200)%4+1
        combined,target,audit=masked_copy(X,np.array(['a','b']*100))
        self.assertEqual(len(combined),2*len(X));self.assertEqual(len(target),len(combined))
        pd.testing.assert_frame_equal(combined.iloc[:len(X)].reset_index(drop=True),X,check_dtype=False)
        for key in ['proto','service','state','Year_Factor']:
            self.assertEqual(combined[key].iloc[len(X):].tolist(),X[key].tolist())
        again,_,_=masked_copy(X,np.array(['a','b']*100))
        pd.testing.assert_frame_equal(combined,again)
        self.assertGreater(audit['masked_cells'],0)

    def test_median_gamma_uses_only_positive_distances(self):
        self.assertAlmostEqual(median_rbf_gamma(np.array([[0.],[0.],[2.]])),.25)
        self.assertEqual(median_rbf_gamma(np.ones((4,2))),1.)

    def test_small_real_estimators_produce_normalized_unseen_predictions(self):
        X=toy();y=np.array(['a','b']*12)
        for strategy in ['balanced_softmax','nystrom_rbf','masked_augmentation']:
            candidate={'strategy':strategy,'parameters':{'n_estimators':3}}
            if strategy=='nystrom_rbf':candidate['parameters']={}
            model=build_next_class(candidate,threads=1).fit(X,y)
            valid=X.iloc[:3].copy();valid['service']='new'
            p=model.predict_proba(valid)
            self.assertEqual(p.shape,(3,2));np.testing.assert_allclose(p.sum(axis=1),1.,atol=1e-6)
        teacher=np.tile([.35,.65],(len(X),1))
        student=build_next_class({'strategy':'soft_distillation','parameters':{'n_estimators':3}},threads=1).fit(X,y,teacher_probabilities=teacher)
        np.testing.assert_allclose(student.predict_proba(X).sum(axis=1),1.)
        self.assertFalse(student.audit_['teacher_is_independent_oof'])

    def test_student_requires_train_teacher_before_any_fit(self):
        with patch('automl_next_class._fit_once',side_effect=AssertionError('No fit')):
            with self.assertRaises(ValueError):build_next_class({'strategy':'soft_distillation'}).fit(toy(),np.array(['a','b']*12))

    def test_bounds_and_single_class(self):
        for parameters in [{'n_estimators':100000},{'device':'cuda'},{'max_depth':33}]:
            with self.assertRaises(ValueError):build_next_class({'strategy':'balanced_softmax','parameters':parameters})
        model=build_next_class({'strategy':'balanced_softmax'}).fit(toy(),np.array(['a']*24))
        np.testing.assert_allclose(model.predict_proba(toy()),1.)

    def test_teacher_cache_missing_never_calls_a_fit(self):
        from experiments.next_classification import read_teacher_probabilities
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            context={'legacy_cache_dir':folder,'teacher_cache_dir':folder,'fit_positions':np.arange(24)}
            with patch('automl_next_class._fit_once',side_effect=AssertionError('Teacher refit forbidden')):
                with self.assertRaises(FileNotFoundError):read_teacher_probabilities(toy(),np.array(['a','b']*12),context)
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_offline_mask_artifacts_record_original_row_lineage(self):
        from experiments.next_classification import evaluate_class
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as folder:
            context={'output':folder,'fit_positions':np.arange(24)+100,'threads':1,'parameters':{'n_estimators':3}}
            result=evaluate_class(60,toy(),np.array(['a','b']*12),toy().iloc[:4],context)
            with np.load(Path(folder)/'training-lineage.npz') as saved:
                np.testing.assert_array_equal(saved['source_fit_positions'],np.tile(context['fit_positions'],2))
                self.assertEqual(saved['is_augmented'].sum(),24)
                self.assertEqual(saved['numeric_mask'].sum(),result['audit']['augmentation']['masked_cells'])
            audit=json.loads((Path(folder)/'training-audit.json').read_text(encoding='utf-8'))
            self.assertFalse(audit['validation_target_used'])
            self.assertEqual(result['probabilities'].shape,(4,2))


if __name__=='__main__':unittest.main()
