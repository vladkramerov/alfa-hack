"""Train-only proper posterior calibration followed by frozen expected-F1 decisions."""
import numpy as np
import pandas as pd
from scipy.special import softmax, logsumexp
from scipy.optimize import minimize
from sprint_decision import MacroF1Decision


def calibrated_ce_gradient(theta, logits, target, penalty=.01):
    theta=np.asarray(theta,float);logits=np.asarray(logits,float);target=np.asarray(target,int)
    inverse_temperature=np.exp(-theta[0]);score=logits*inverse_temperature+theta[1:]
    loss=np.mean(logsumexp(score,axis=1)-score[np.arange(len(target)),target])+penalty*np.dot(theta,theta)/2
    residual=softmax(score,axis=1);residual[np.arange(len(target)),target]-=1;residual/=len(target)
    grad=np.r_[np.sum(residual*(-logits*inverse_temperature)),residual.sum(axis=0)]+penalty*theta
    return float(loss),grad


class CalibratedMacroDecision:
    def __init__(self,penalty=.01,iterations=100):
        self.penalty,self.iterations=float(penalty),int(iterations)
    def configuration(self):
        return {'strategy':'proper_posterior_expected_macro_decision','version':1,'calibration':'unweighted_ce_scalar_temperature_and_class_bias',
                'penalty':self.penalty,'iterations':self.iterations,'log_temperature_bounds':[-2.3,2.3],
                'bias_bounds':[-5.,5.],'probability_clip':1e-6,'decision':MacroF1Decision().configuration()}
    def _teacher(self,p,rows):
        if isinstance(p,pd.DataFrame):
            if not p.columns.is_unique or set(p.columns)!=set(self.classes_):raise ValueError('Teacher class order mismatch')
            p=p.loc[:,self.classes_]
        p=np.asarray(p,float)
        if p.shape!=(rows,len(self.classes_)) or not np.isfinite(p).all() or np.any(p<0) or np.any(p.sum(axis=1)<=0):raise ValueError('Invalid teacher')
        p=p/p.sum(axis=1,keepdims=True)
        return np.log(np.clip(p,1e-6,1))
    def _probabilities(self,p,rows):
        logits=self._teacher(p,rows)
        return softmax(logits*np.exp(-self.theta_[0])+self.theta_[1:],axis=1)
    def fit(self,X,y,teacher_probabilities=None):
        y=np.asarray(y)
        if y.ndim!=1 or len(y)!=len(X) or not len(y) or pd.isna(y).any():raise ValueError('Aligned nonempty labels required')
        self.classes_,target=np.unique(y,return_inverse=True)
        logits=self._teacher(teacher_probabilities,len(X));initial=np.zeros(len(self.classes_)+1)
        objective=lambda theta:calibrated_ce_gradient(theta,logits,target,self.penalty)
        optimized=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=[(-2.3,2.3)]+[(-5.,5.)]*len(self.classes_),
                           options={'maxiter':self.iterations,'ftol':1e-10})
        self.theta_=optimized.x if np.isfinite(optimized.fun) and optimized.fun<=objective(initial)[0] else initial
        calibrated=self._probabilities(teacher_probabilities,len(X))
        self.decision_=MacroF1Decision().fit(X,y,teacher_probabilities=calibrated)
        self.audit_={'fit_rows':len(X),'statistics_source':'training_only','initial_training_ce':objective(initial)[0],
                     'final_training_ce':objective(self.theta_)[0],'optimizer_success':bool(optimized.success),
                     'optimizer_iterations':int(optimized.nit),'theta':self.theta_.tolist(),'decision':self.decision_.audit_,
                     'teacher_oof_provenance_is_callers_responsibility':True,'query_statistics':False}
        return self
    def predict_proba(self,X,teacher_probabilities=None):
        return self.decision_.predict_proba(X,teacher_probabilities=self._probabilities(teacher_probabilities,len(X)))
    def predict(self,X,teacher_probabilities=None):
        return self.classes_[self.predict_proba(X,teacher_probabilities).argmax(axis=1)]
