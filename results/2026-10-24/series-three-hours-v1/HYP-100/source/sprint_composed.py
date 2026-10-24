"""H100 frozen H78 decisions followed by new empirical probability offsets."""
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.special import softmax
from sprint_empirical import EmpiricalMacroF1Offsets


class ComposedEmpiricalDecision(EmpiricalMacroF1Offsets):
    def __init__(self,state):
        self.state=deepcopy(state)
    def configuration(self):
        c=super().configuration()
        c.update(hypothesis='HYP-100',strategy='frozen_H78_plus_empirical_probability_offsets',input_state=deepcopy(self.state),input_state_refitted=False,
                 training_teacher='compatible_base_teacher_through_frozen_H78_state',outer_teacher='H48_through_frozen_H78_state',
                 H48_meta_state_used=True,training_meta_is_independent_OOF=False)
        return c
    def _transform(self,p,rows):
        classes=np.asarray(self.state.get('classes'));gain=np.asarray(self.state.get('gain'),float)
        offset=np.asarray(self.state.get('offset'),float);theta=np.asarray(self.state.get('theta'),float)
        if (classes.ndim!=1 or len(classes)<2 or not np.array_equal(classes,np.unique(classes)) or gain.shape!=classes.shape or offset.shape!=classes.shape
                or theta.shape!=(len(classes)+1,) or not np.isfinite(np.r_[gain,offset,theta]).all() or np.any(gain<=0)):
            raise ValueError('Valid sorted frozen H78 state required')
        if isinstance(p,pd.DataFrame):
            if not p.columns.is_unique or set(p.columns)!=set(classes):raise ValueError('Teacherclasses mismatch')
            p=p.loc[:,classes]
        p=np.asarray(p,float)
        if p.shape!=(rows,len(classes)) or not np.isfinite(p).all() or np.any(p<0) or np.any(p.sum(axis=1)<=0):raise ValueError('Invalidteacher')
        p=p/p.sum(axis=1,keepdims=True)
        calibrated=softmax(np.log(np.clip(p,1e-6,1))*np.exp(-theta[0])+theta[1:],axis=1)
        return softmax(calibrated*gain+offset,axis=1)
    def fit(self,X,y,teacher_probabilities=None):
        if not np.array_equal(np.unique(y),self.state.get('classes')):raise ValueError('Frozen classes differfromtraining')
        return super().fit(X,y,self._transform(teacher_probabilities,len(X)))
    def predict_proba(self,X,teacher_probabilities=None):
        return super().predict_proba(X,self._transform(teacher_probabilities,len(X)))
