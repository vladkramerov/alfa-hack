"""H98 direct empirical macro-F1 search for additive probability offsets.

No gradient, learner fitting, query statistics, or log-probability multipliers.
Caller must supply independently cross-fitted training teacher probabilities.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax


def _macro_f1(y, prediction, classes):
    truth=np.bincount(y,minlength=classes)
    predicted=np.bincount(prediction,minlength=classes)
    tp=np.bincount(y[y==prediction],minlength=classes)
    denominator=truth+predicted
    return float(np.mean(np.divide(2.*tp,denominator,out=np.zeros(classes),where=denominator>0)))


def best_class_offset(probabilities, labels, offsets, class_index, tolerance=1e-12):
    """Exact O(NK) count curve after one margin sort for a frozen coordinate.

    Only open intervals between distinct margin groups are candidates. Floating
    candidates whose actual score addition produces an unexpected boundary tie
    are rejected; no tied group is split to improve the training metric.
    """
    p,y,b=np.asarray(probabilities,float),np.asarray(labels),np.asarray(offsets,float)
    if (p.ndim!=2 or not len(p) or p.shape[1]<2 or not np.isfinite(p).all() or np.any(p<0)
            or not np.allclose(p.sum(1),1.,atol=1e-12,rtol=1e-12)
            or y.shape!=(len(p),) or y.dtype.kind not in 'iu' or np.any(y<0) or np.any(y>=p.shape[1])
            or b.shape!=(p.shape[1],) or not np.isfinite(b).all()
            or not isinstance(class_index,(int,np.integer)) or not 0<=class_index<p.shape[1]
            or not np.isfinite(tolerance) or tolerance<=0):
        raise ValueError('Expected normalized probabilities, integer labels, finite offsets and valid coordinate')
    n,k=p.shape;c=int(class_index)
    current_prediction=np.argmax(p+b,axis=1)
    current=_macro_f1(y,current_prediction,k)
    other_scores=p+b
    other_scores[:,c]=-np.inf
    other=np.argmax(other_scores,axis=1)
    margin=other_scores[np.arange(n),other]-p[:,c]
    order=np.argsort(margin,kind='stable')
    sorted_margin=margin[order];sy=y[order];so=other[order]
    ends=np.r_[np.flatnonzero(sorted_margin[1:]!=sorted_margin[:-1]),n-1]
    boundary=sorted_margin[ends]
    prefix=np.r_[0,ends+1]
    thresholds=np.r_[np.nextafter(boundary[0],-np.inf),
        boundary[:-1]+(boundary[1:]-boundary[:-1])/2,np.nextafter(boundary[-1],np.inf)]
    usable=np.isfinite(thresholds)
    if len(boundary)>1:
        usable[1:-1]&=(thresholds[1:-1]>boundary[:-1])&(thresholds[1:-1]<boundary[1:])
    truth=np.bincount(y,minlength=k)
    base_count=np.bincount(other,minlength=k)
    base_tp=np.bincount(other[other==y],minlength=k)
    curve=np.zeros(len(prefix))
    for j in range(k):
        if j==c:
            q=prefix
            tp=np.r_[0,np.cumsum(sy==c)][prefix]
        else:
            q=base_count[j]-np.r_[0,np.cumsum(so==j)][prefix]
            tp=base_tp[j]-np.r_[0,np.cumsum((so==j)&(sy==j))][prefix]
        denominator=truth[j]+q
        curve+=np.divide(2.*tp,denominator,out=np.zeros(len(prefix)),where=denominator>0)/k
    rejected=0
    for index in np.argsort(-curve,kind='stable'):
        if curve[index]<=current+tolerance:
            break
        if not usable[index]:
            continue
        proposed=b.copy();proposed[c]=thresholds[index]
        actual=np.argmax(p+proposed,axis=1)
        ideal=other.copy();ideal[margin<thresholds[index]]=c
        if not np.array_equal(actual,ideal):
            rejected+=1
            continue
        actual_score=_macro_f1(y,actual,k)
        if abs(actual_score-curve[index])>1e-12:
            raise RuntimeError('Exact F1 curve disagrees with actual candidate prediction')
        if actual_score>current+tolerance:
            return {'offset':float(thresholds[index]),'score':actual_score,'improved':True,
                    'candidate_intervals':len(thresholds),'float_ambiguous_rejected':rejected}
    return {'offset':float(b[c]),'score':current,'improved':False,
            'candidate_intervals':len(thresholds),'float_ambiguous_rejected':rejected}


class EmpiricalMacroF1Offsets:
    """Freeze p_k+b_k decisions using actual training labels and honest OOF p."""
    def configuration(self):
        return {'hypothesis':'HYP-098','strategy':'exact_empirical_macro_f1_probability_offsets','version':1,
                'passes':3,'class_order':'ascending_training_count_then_sorted_class_index',
                'initialization':'zero_offsets','objective':'actual_training_labels_macro_f1',
                'coordinate_search':'exact_between_distinct_margin_groups_plus_outside_endpoints',
                'strict_improvement_tolerance':1e-12,'float_ambiguous_candidates':'reject',
                'training_teacher':'reused_H88_grouped_oof_RAW',
                'outer_teacher':'original_grouped_oof_RAW_with_H11_query',
                'H48_meta_state_used':False,'query_statistics':False,
                'output':'softmax(probability_plus_frozen_offset)_not_calibrated'}

    def _teacher(self,p,rows):
        if p is None:
            raise ValueError('Explicit honest OOF teacher probabilities required')
        if isinstance(p,pd.DataFrame) and list(p.columns)!=list(self.classes_):
            raise ValueError('Teacher columns must match sorted training classes')
        p=np.asarray(p,float)
        if p.shape!=(rows,len(self.classes_)) or not np.isfinite(p).all() or np.any(p<0):
            raise ValueError('Invalid teacher probabilities or class count')
        scale=p.max(axis=1,keepdims=True)
        if np.any(scale<=0):raise ValueError('Teacher rows require positive probability mass')
        p=p/scale
        return p/p.sum(axis=1,keepdims=True)

    def fit(self,X,y,teacher_probabilities=None):
        target=np.asarray(y)
        if target.ndim!=1 or len(target)!=len(X) or not len(target) or pd.isna(target).any():
            raise ValueError('Complete nonempty training labels matching X required')
        self.classes_,encoded,counts=np.unique(target,return_inverse=True,return_counts=True)
        if len(self.classes_)<2:raise ValueError('Need at least two training classes')
        p=self._teacher(teacher_probabilities,len(X))
        self.offset_=np.zeros(len(self.classes_))
        visits=np.lexsort((np.arange(len(counts)),counts))
        score=_macro_f1(encoded,np.argmax(p,axis=1),len(counts))
        history=[score];steps=[];passes=0
        for pass_number in range(3):
            changed=False;passes+=1
            for c in visits:
                candidate=best_class_offset(p,encoded,self.offset_,int(c))
                if candidate['improved']:
                    self.offset_[c]=candidate['offset'];score=candidate['score']
                    history.append(score);changed=True
                    steps.append({'pass':pass_number+1,'class':self.classes_[c].item() if isinstance(self.classes_[c],np.generic) else self.classes_[c],**candidate})
            if not changed:break
        final=_macro_f1(encoded,np.argmax(p+self.offset_,axis=1),len(counts))
        if final<history[0]-1e-12 or abs(final-history[-1])>1e-12:
            raise RuntimeError('Final empirical decisions lost training F1')
        self.audit_={'fit_rows':len(X),'classes':self.classes_.tolist(),'class_counts':counts.tolist(),
            'offset':self.offset_.tolist(),'class_visit_order':self.classes_[visits].tolist(),
            'objective':'actual_training_labels_macro_f1','initial_training_macro_f1':history[0],
            'selected_training_macro_f1':final,'training_macro_f1_history':history,
            'accepted_steps':steps,'passes_completed':passes,'query_statistics':False,
            'training_macro_f1_is_independent_evaluation':False,'configuration':self.configuration()}
        return self

    def predict_proba(self,X,teacher_probabilities=None):
        if not hasattr(self,'offset_'):raise ValueError('Fit empirical offsets before prediction')
        p=self._teacher(teacher_probabilities,len(X))
        return softmax(p+self.offset_,axis=1)

    def predict(self,X,teacher_probabilities=None):
        return self.classes_[self.predict_proba(X,teacher_probabilities).argmax(axis=1)]
