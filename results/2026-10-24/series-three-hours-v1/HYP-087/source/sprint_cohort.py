"""H87: unlabeled-query expected macro-F1 decisions with frozen calibration."""
from copy import deepcopy
import numpy as np
import pandas as pd
from scipy.special import softmax
from sprint_decision import expected_macro_f1_gradient


class CohortMacroDecision:
    def __init__(self,calibration_state=None,iterations=25,damping=.5,epsilon=1e-8):
        self.calibration_state=deepcopy(calibration_state)
        self.iterations,self.damping,self.epsilon=int(iterations),float(damping),float(epsilon)
    def configuration(self):
        return {'strategy':'unlabeled_cohort_expected_macro_decision','version':1,'iterations':self.iterations,
                'damping':self.damping,'epsilon':self.epsilon,'calibration_state':deepcopy(self.calibration_state),
                'query_labels_used':False,'query_prior':'mean_calibrated_query_posterior',
                'batch_invariant':False,'query_statistics':True,'initialization':'teacher_argmax','selection':'best_query_expected_objective'}
    def _p(self,p,rows):
        if isinstance(p,pd.DataFrame):
            if set(p.columns)!=set(self.classes_) or not p.columns.is_unique:raise ValueError('Teacherclasses mismatch')
            p=p.loc[:,self.classes_]
        p=np.asarray(p,float)
        if p.shape!=(rows,len(self.classes_)) or not np.isfinite(p).all() or np.any(p<0) or np.any(p.sum(axis=1)<=0):raise ValueError('Invalidteacher')
        p=p/p.sum(axis=1,keepdims=True)
        return softmax(np.log(np.clip(p,1e-6,1))*np.exp(-self.theta_[0])+self.theta_[1:],axis=1)
    def fit(self,X,y,teacher_probabilities=None):
        y=np.asarray(y)
        if y.shape!=(len(X),) or not len(y) or pd.isna(y).any():raise ValueError('Invalidtraininglabels')
        self.classes_=np.unique(y)
        self.theta_=np.asarray((self.calibration_state or {}).get('theta'),float)
        if self.theta_.shape!=(len(self.classes_)+1,) or not np.isfinite(self.theta_).all():raise ValueError('Explicit frozen compatiblecalibration required')
        self._p(teacher_probabilities,len(X))
        self.audit_={'fit_rows':len(X),'classes':self.classes_.tolist(),'learner_fits':0,'calibration_refitted':False,
                     'configuration':self.configuration(),'query_target_used':False}
        return self
    def predict_proba(self,X,teacher_probabilities=None):
        p=self._p(teacher_probabilities,len(X));k=len(self.classes_)
        if not len(p):return p
        prior=p.mean(axis=0);gain=np.ones(k);offset=np.zeros(k)
        def moments(a,b):
            decision=np.argmax(p*a+b,axis=1)
            return (np.bincount(decision,minlength=k)/len(p),
                    np.bincount(decision,weights=p[np.arange(len(p)),decision],minlength=k)/len(p))
        q,t=moments(gain,offset);score,dt,dq=expected_macro_f1_gradient(prior,q,t,self.epsilon)
        first=best=score;bestgain,bestoffset=gain.copy(),offset.copy();selected=0
        for iteration in range(1,self.iterations+1):
            gain=(1-self.damping)*gain+self.damping*k*dt
            offset=(1-self.damping)*offset+self.damping*k*dq
            q,t=moments(gain,offset);score,dt,dq=expected_macro_f1_gradient(prior,q,t,self.epsilon)
            if score>best:best,bestgain,bestoffset,selected=score,gain.copy(),offset.copy(),iteration
        self.last_prediction_audit_={'query_rows':len(X),'query_labels_used':False,'query_prior':prior.tolist(),
                                    'initial_expected_macro_f1':first,'selected_expected_macro_f1':best,
                                    'selected_iteration':selected,'gain':bestgain.tolist(),'offset':bestoffset.tolist()}
        return softmax(p*bestgain+bestoffset,axis=1)
    def predict(self,X,teacher_probabilities=None):
        return self.classes_[self.predict_proba(X,teacher_probabilities).argmax(axis=1)]
