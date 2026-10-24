"""Behavioral checks for the new experiment protocol, without public model fits."""
import importlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd


class NovelModelsTests(unittest.TestCase):
    def api(self): return importlib.import_module('experiments.novel_models')

    def test_group_weights_do_not_depend_on_labels_or_normalization(self):
        X=pd.DataFrame({'a':[1,1,2,3,3,3],'b':['x','x','y','z','z','z']})
        weights=self.api().inverse_group_weights(X)
        np.testing.assert_allclose(weights,[.5,.5,1,1/3,1/3,1/3])
        self.assertAlmostEqual(float(weights.sum()),3.)

    def test_hierarchy_aligns_global_class_columns_and_preserves_probability_mass(self):
        classes=np.array(['Attack-A','Normal','Attack-B'])
        probabilities=self.api().hierarchy_probabilities(np.array([.8,.3]),np.array([[.25,.75],[.7,.3]]),np.array(['Attack-B','Attack-A']),classes,'Normal')
        np.testing.assert_allclose(probabilities,[[.6,.2,.2],[.09,.7,.21]])
        np.testing.assert_allclose(probabilities.sum(axis=1),1)
        with self.assertRaises(ValueError): self.api().hierarchy_probabilities(np.array([.8]),np.array([[.2,.8]]),['Attack-B','unknown'],classes,'Normal')

    def test_target_log_inverse_keeps_evaluation_in_original_units(self):
        api=self.api(); y=np.array([0.,1.,10.,100.])
        encoded=api.transform_target(y,'log1p')
        np.testing.assert_allclose(api.inverse_target(encoded,'log1p'),y,atol=1e-12)
        with self.assertRaises(ValueError): api.transform_target(np.array([-2.,1.]),'log1p')

    def test_fixed_plan_contains_eleven_new_evaluations(self):
        plans=self.api().experiment_plan()
        self.assertEqual(len(plans),11)
        self.assertEqual(len({(p['hypothesis'],p['task']) for p in plans}),11)
        self.assertEqual({p['hypothesis'] for p in plans},{f'HYP-{n:03d}' for n in range(11,21)})
        self.assertEqual(sum(p['hypothesis']=='HYP-011' for p in plans),2)

    def test_fingerprint_ignores_seed_and_display_name_but_distinguishes_mechanism(self):
        api=self.api(); a=api.experiment_plan()[0]
        b={**a,'hypothesis':'renamed','parameters':{**a['parameters'],'random_state':77}}
        self.assertEqual(api.semantic_key(a),api.semantic_key(b))
        c={**a,'transform':'log1p_target'}
        self.assertNotEqual(api.semantic_key(a),api.semantic_key(c))

    def test_completed_or_started_hypothesis_cannot_be_registered_again(self):
        api=importlib.import_module('experiments.run_novel_hypotheses')
        candidate=self.api().experiment_plan()[0]
        with tempfile.TemporaryDirectory(dir=Path.cwd()/'tmp') as root:
            directory=Path(root); (directory/'HYPOTHESES.md').write_text('HYP-011 planned',encoding='utf-8')
            api.claim_attempt(directory,candidate,{'train.csv':'a'},'fixedsplit')
            with self.assertRaises(RuntimeError): api.claim_attempt(directory,{**candidate,'hypothesis':'renamed'}, {'train.csv':'a'},'fixedsplit')

    def test_new_branches_fit_toy_data_and_predict_unknown_categories(self):
        api=self.api(); signal=np.arange(75,dtype=float)/10
        X=pd.DataFrame({'signal':signal,'proto':['tcp','udp','tcp']*25,'service':['http','dns','ftp']*25,'state':['FIN','INT','CON']*25})
        future=X.iloc[:5].copy(); future['service']='never-seen'; future.loc[0,'signal']=np.nan
        for original in api.experiment_plan():
            candidate={**original,'parameters':{**original['parameters'],'n_estimators':6}}
            if candidate['family']=='lightgbm': candidate['parameters']['min_child_samples']=2
            y=1+signal**2 if candidate['task']=='regression' else np.tile(['Normal','Attack-a','Attack-b'],25)
            with self.subTest(hypothesis=candidate['hypothesis'],task=candidate['task']):
                model=api.fit_experiment(candidate,X,y)
                prediction,probabilities,classes=api.predictions(candidate,model,future)
                self.assertEqual(len(prediction),5)
                if probabilities is not None:
                    self.assertEqual(probabilities.shape,(5,3)); self.assertTrue(np.isfinite(probabilities).all())
                else: self.assertTrue(np.isfinite(prediction).all())


if __name__=='__main__': unittest.main()
