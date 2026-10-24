from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest
from automl_models import canonical_candidate, build_pipeline
from automl_neural import NeuralComplement


def test_neural_scope_and_dispatch():
    candidate={'family':'extra_trees','panel_time_column':'clock','neural_complement':True}
    assert isinstance(build_pipeline('regression',candidate),NeuralComplement)
    with pytest.raises(ValueError):build_pipeline('classification',candidate)
    with pytest.raises(ValueError):canonical_candidate({'family':'linear','neural_complement':True})


def test_weight_fit_is_prefix_only_and_refit_uses_all_rows():
    seen=[]
    class Base:
        feature_code_=''
        audit_={}
        def fit(self,X,y):
            seen.append(('base',len(X),float(X.clock.max())))
            return self
        def predict(self,X):return np.zeros(len(X))
    class Neural:
        audit_={}
        def fit(self,X,y):
            seen.append(('neural',len(X),float(X.clock.max())))
            return self
        def predict(self,X):return X.measure.to_numpy()*2
    X=pd.DataFrame({'clock':np.repeat([1,2,3],600),'measure':np.tile(np.arange(600)+1,3)})
    with patch('automl_models.build_pipeline',side_effect=lambda *a:Base()), \
         patch('automl_torch_tabm.TorchTabM',side_effect=lambda **k:Neural()):
        model=NeuralComplement({'family':'linear','panel_time_column':'clock','neural_complement':True}).fit(X,X.measure*2)
    assert seen==[('base',1200,2.),('neural',1200,2.),('base',1800,3.),('neural',1800,3.)]
    assert model.weight_==pytest.approx(1)
    assert model.audit_['neural_complement']['query_targets_used'] is False
    np.testing.assert_array_equal(model.predict(X.iloc[:5]),X.measure.iloc[:5]*2)
