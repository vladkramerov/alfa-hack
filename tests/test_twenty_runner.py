import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

class TwentyCommonTests(unittest.TestCase):
    def test_inverse_target_and_label_mapping(self):
        from experiments.twenty_common import make_model
        X=pd.DataFrame({'x':np.arange(30,dtype=float),'c':['a','b']*15})
        m=make_model('regression',parameters={'n_estimators':3}).fit(X,np.arange(30.)+1)
        self.assertTrue(np.isfinite(m.predict(X)).all())
        c=make_model('multiclass',parameters={'n_estimators':3}).fit(X,np.array(['rare','other']*15))
        self.assertEqual(set(c.classes_),{'rare','other'})
        self.assertTrue(np.allclose(c.predict_proba(X).sum(axis=1),1,atol=1e-6))
    def test_fit_registrar_rejects_duplicate_even_seed_change(self):
        from experiments.twenty_common import make_model
        X=pd.DataFrame({'x':np.arange(30.)}); y=np.arange(30.)
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as d, patch.dict(os.environ,{'SPEC006_RUN_DIR':d}):
            make_model('regression',parameters={'n_estimators':2}).fit(X,y)
            with self.assertRaises(RuntimeError):
                make_model('regression',parameters={'n_estimators':2,'random_state':123}).fit(X,y)
            self.assertEqual(len((Path(d)/'model-fit-attempts.jsonl').read_text().splitlines()),1)
    def test_plan_covers_twenty_and_twenty_three_evaluations(self):
        from experiments.run_twenty_hypotheses import experiment_plan
        plan=experiment_plan()
        self.assertEqual(len(plan),23)
        self.assertEqual({p['number'] for p in plan},set(range(22,42)))
    def test_legacy_inner_matches_saved_baseline_numeric_representation(self):
        from experiments.twenty_common import make_model
        from automl_models import Encoder
        X=pd.DataFrame({'x':[0.,100.,np.nan,50.],'c':['a','b','a','b']})
        model=make_model('regression',policy='legacy',parameters={'n_estimators':2}).fit(X,np.arange(4.))
        old=Encoder(scale=False).fit(X)
        np.testing.assert_array_equal(model.encoder_.transform(X),old.transform(X))

if __name__=='__main__': unittest.main()
