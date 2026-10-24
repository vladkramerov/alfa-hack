import unittest
import numpy as np
import pandas as pd
from scipy.special import softmax


class OrderedTests(unittest.TestCase):
    def test_ordering_and_teacher_margin_once(self):
        from sprint_ordered import OrderedHonestCorrector
        model=OrderedHonestCorrector(iterations=2)
        self.assertEqual(model.parameters['boosting_type'],'Ordered')
        self.assertEqual(model.configuration()['hypothesis'],'HYP-103')
        X=pd.DataFrame({'n':np.arange(60,dtype=float),'proto':['tcp','udp']*30})
        y=np.array(['a','b','c']*20)
        p=np.tile([.2,.3,.5],(60,1))
        model.fit(X,y,p)
        from catboost import Pool
        logp=model._teacher(p,len(X))
        frame=model._features(X,logp)
        raw=model.model_.predict(Pool(frame,cat_features=model.categorical_),prediction_type='RawFormulaVal')
        np.testing.assert_allclose(model.predict_proba(X,p),softmax(raw+logp,axis=1))
        np.testing.assert_allclose(model.predict_proba(X.iloc[:7],p[:7]),model.predict_proba(X,p)[:7])

    def test_frozen_only_boosting_scheme_changes(self):
        from sprint_ordered import OrderedHonestCorrector
        from sprint_honest import HonestContextCorrector
        ordered=OrderedHonestCorrector()
        plain=HonestContextCorrector()
        params=ordered.parameters.copy();params.pop('boosting_type')
        self.assertEqual(params,plain.parameters)


if __name__=='__main__':unittest.main()
